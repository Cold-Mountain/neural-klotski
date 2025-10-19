"""
Rendering Optimization for Neural-Klotski Visualization

Performance optimization techniques including viewport culling,
level-of-detail rendering, caching, and batch processing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from enum import Enum
import time
import math
import weakref
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.visualization.layout.coordinate_system import Point2D, BoundingBox
from neural_klotski.visualization.rendering.visual_elements import VisualElement, Color
from neural_klotski.visualization.rendering.render_context import DrawingPrimitive, PrimitiveType


class LODLevel(Enum):
    """Level of detail levels"""
    FULL = "full"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class CullingResult(Enum):
    """Viewport culling results"""
    VISIBLE = "visible"
    PARTIALLY_VISIBLE = "partially_visible"
    OUTSIDE = "outside"
    CLIPPED = "clipped"


@dataclass
class PerformanceSettings:
    """Performance optimization settings"""
    # Viewport culling
    enable_culling: bool = True
    culling_margin: float = 50.0  # Extra margin around viewport

    # Level of detail
    enable_lod: bool = True
    lod_distances: Dict[LODLevel, float] = field(default_factory=lambda: {
        LODLevel.FULL: 1.0,
        LODLevel.HIGH: 0.5,
        LODLevel.MEDIUM: 0.25,
        LODLevel.LOW: 0.1,
        LODLevel.MINIMAL: 0.05
    })

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: float = 5.0
    max_cache_entries: int = 1000

    # Batching
    enable_batching: bool = True
    max_batch_size: int = 500
    batch_by_material: bool = True

    # Frame rate management
    target_fps: int = 60
    adaptive_quality: bool = True
    min_frame_time_ms: float = 16.67  # 60 FPS

    # Memory management
    gc_interval_frames: int = 300  # Garbage collect every 5 seconds at 60fps
    max_memory_mb: float = 512.0


@dataclass
class RenderStats:
    """Rendering performance statistics"""
    # Timing
    frame_time_ms: float = 0.0
    render_time_ms: float = 0.0
    cull_time_ms: float = 0.0
    lod_time_ms: float = 0.0

    # Counts
    total_elements: int = 0
    visible_elements: int = 0
    culled_elements: int = 0
    cached_elements: int = 0
    batched_primitives: int = 0

    # Memory
    cache_memory_mb: float = 0.0
    total_memory_mb: float = 0.0

    # Quality
    current_lod_level: LODLevel = LODLevel.FULL
    quality_factor: float = 1.0

    def reset(self) -> None:
        """Reset statistics for new frame"""
        self.frame_time_ms = 0.0
        self.render_time_ms = 0.0
        self.cull_time_ms = 0.0
        self.lod_time_ms = 0.0
        self.total_elements = 0
        self.visible_elements = 0
        self.culled_elements = 0
        self.cached_elements = 0
        self.batched_primitives = 0


class ViewportCuller:
    """
    Viewport culling system for efficient rendering.

    Eliminates elements outside the visible viewport to
    reduce rendering overhead.
    """

    def __init__(self, settings: PerformanceSettings):
        """Initialize viewport culler"""
        self.settings = settings

        # Culling statistics
        self.cull_stats = {
            'total_checks': 0,
            'elements_culled': 0,
            'elements_visible': 0,
            'cull_time_ms': 0.0
        }

        # Spatial partitioning for faster culling
        self.spatial_grid: Dict[Tuple[int, int], List[VisualElement]] = {}
        self.grid_cell_size = 100.0

    def cull_elements(self,
                     elements: List[VisualElement],
                     viewport_bounds: BoundingBox,
                     zoom_level: float = 1.0) -> Tuple[List[VisualElement], List[VisualElement]]:
        """
        Cull elements against viewport.

        Args:
            elements: Elements to cull
            viewport_bounds: Current viewport bounds
            zoom_level: Current zoom level

        Returns:
            Tuple of (visible_elements, culled_elements)
        """
        if not self.settings.enable_culling:
            return elements, []

        cull_start = time.perf_counter()

        # Expand viewport by margin
        margin = self.settings.culling_margin / zoom_level
        expanded_viewport = viewport_bounds.expand(margin)

        visible_elements = []
        culled_elements = []

        for element in elements:
            self.cull_stats['total_checks'] += 1

            # Get element bounds
            element_bounds = element.get_bounds()

            # Perform culling test
            cull_result = self._test_element_visibility(element_bounds, expanded_viewport)

            if cull_result in [CullingResult.VISIBLE, CullingResult.PARTIALLY_VISIBLE]:
                visible_elements.append(element)
                self.cull_stats['elements_visible'] += 1
            else:
                culled_elements.append(element)
                self.cull_stats['elements_culled'] += 1

        # Update timing
        cull_time = (time.perf_counter() - cull_start) * 1000
        self.cull_stats['cull_time_ms'] += cull_time

        return visible_elements, culled_elements

    def _test_element_visibility(self, element_bounds: BoundingBox, viewport_bounds: BoundingBox) -> CullingResult:
        """Test if element is visible in viewport"""
        # Check for complete outside
        if (element_bounds.max_x < viewport_bounds.min_x or
            element_bounds.min_x > viewport_bounds.max_x or
            element_bounds.max_y < viewport_bounds.min_y or
            element_bounds.min_y > viewport_bounds.max_y):
            return CullingResult.OUTSIDE

        # Check for complete inside
        if (element_bounds.min_x >= viewport_bounds.min_x and
            element_bounds.max_x <= viewport_bounds.max_x and
            element_bounds.min_y >= viewport_bounds.min_y and
            element_bounds.max_y <= viewport_bounds.max_y):
            return CullingResult.VISIBLE

        # Partially visible
        return CullingResult.PARTIALLY_VISIBLE

    def update_spatial_grid(self, elements: List[VisualElement]) -> None:
        """Update spatial grid for faster culling"""
        self.spatial_grid.clear()

        for element in elements:
            bounds = element.get_bounds()
            grid_coords = self._get_grid_coordinates(bounds)

            for coord in grid_coords:
                if coord not in self.spatial_grid:
                    self.spatial_grid[coord] = []
                self.spatial_grid[coord].append(element)

    def _get_grid_coordinates(self, bounds: BoundingBox) -> List[Tuple[int, int]]:
        """Get grid coordinates that bounds intersect"""
        coords = []

        min_grid_x = int(bounds.min_x // self.grid_cell_size)
        max_grid_x = int(bounds.max_x // self.grid_cell_size)
        min_grid_y = int(bounds.min_y // self.grid_cell_size)
        max_grid_y = int(bounds.max_y // self.grid_cell_size)

        for grid_x in range(min_grid_x, max_grid_x + 1):
            for grid_y in range(min_grid_y, max_grid_y + 1):
                coords.append((grid_x, grid_y))

        return coords

    def get_statistics(self) -> Dict[str, Any]:
        """Get culling statistics"""
        return self.cull_stats.copy()


class LevelOfDetail:
    """
    Level of detail (LOD) system for adaptive quality rendering.

    Adjusts rendering quality based on zoom level, performance,
    and distance from viewport center.
    """

    def __init__(self, settings: PerformanceSettings):
        """Initialize LOD system"""
        self.settings = settings
        self.current_lod = LODLevel.FULL

        # LOD configuration for different element types
        self.lod_config = {
            'labels': {
                LODLevel.FULL: {'visible': True, 'font_scale': 1.0},
                LODLevel.HIGH: {'visible': True, 'font_scale': 0.9},
                LODLevel.MEDIUM: {'visible': True, 'font_scale': 0.7},
                LODLevel.LOW: {'visible': False, 'font_scale': 0.5},
                LODLevel.MINIMAL: {'visible': False, 'font_scale': 0.3}
            },
            'indicators': {
                LODLevel.FULL: {'visible': True, 'size_scale': 1.0},
                LODLevel.HIGH: {'visible': True, 'size_scale': 0.8},
                LODLevel.MEDIUM: {'visible': True, 'size_scale': 0.6},
                LODLevel.LOW: {'visible': False, 'size_scale': 0.4},
                LODLevel.MINIMAL: {'visible': False, 'size_scale': 0.2}
            },
            'shadows': {
                LODLevel.FULL: {'enabled': True, 'quality': 1.0},
                LODLevel.HIGH: {'enabled': True, 'quality': 0.8},
                LODLevel.MEDIUM: {'enabled': True, 'quality': 0.6},
                LODLevel.LOW: {'enabled': False, 'quality': 0.4},
                LODLevel.MINIMAL: {'enabled': False, 'quality': 0.2}
            }
        }

        # Performance monitoring
        self.lod_stats = {
            'lod_changes': 0,
            'elements_processed': 0,
            'quality_adjustments': 0
        }

    def calculate_lod_level(self,
                           zoom_level: float,
                           performance_factor: float = 1.0,
                           element_distance: float = 0.0) -> LODLevel:
        """
        Calculate appropriate LOD level.

        Args:
            zoom_level: Current zoom level (1.0 = normal)
            performance_factor: Performance factor (0.0 to 1.0, lower = worse performance)
            element_distance: Distance from viewport center (normalized)

        Returns:
            Appropriate LOD level
        """
        if not self.settings.enable_lod:
            return LODLevel.FULL

        # Base LOD from zoom level
        zoom_factor = min(1.0, zoom_level)

        # Adjust for performance
        adjusted_factor = zoom_factor * performance_factor

        # Adjust for distance (elements farther from center get lower LOD)
        distance_factor = max(0.0, 1.0 - element_distance)
        final_factor = adjusted_factor * distance_factor

        # Map to LOD level
        if final_factor >= 0.8:
            return LODLevel.FULL
        elif final_factor >= 0.6:
            return LODLevel.HIGH
        elif final_factor >= 0.4:
            return LODLevel.MEDIUM
        elif final_factor >= 0.2:
            return LODLevel.LOW
        else:
            return LODLevel.MINIMAL

    def apply_lod_to_element(self, element: VisualElement, lod_level: LODLevel) -> None:
        """Apply LOD settings to visual element"""
        self.lod_stats['elements_processed'] += 1

        # Determine element type
        element_type = self._get_element_type(element)

        if element_type in self.lod_config:
            config = self.lod_config[element_type].get(lod_level, {})

            # Apply visibility
            if 'visible' in config:
                element.visible = config['visible']

            # Apply scaling
            if 'size_scale' in config and hasattr(element, 'size'):
                original_size = getattr(element, '_original_size', element.size)
                if not hasattr(element, '_original_size'):
                    element._original_size = element.size

                scale = config['size_scale']
                element.size = Point2D(original_size.x * scale, original_size.y * scale)

            # Apply font scaling for text elements
            if 'font_scale' in config and hasattr(element, 'font_size'):
                original_font_size = getattr(element, '_original_font_size', element.font_size)
                if not hasattr(element, '_original_font_size'):
                    element._original_font_size = element.font_size

                scale = config['font_scale']
                element.font_size = original_font_size * scale

            self.lod_stats['quality_adjustments'] += 1

    def set_global_lod(self, lod_level: LODLevel) -> None:
        """Set global LOD level"""
        if lod_level != self.current_lod:
            self.current_lod = lod_level
            self.lod_stats['lod_changes'] += 1

    def _get_element_type(self, element: VisualElement) -> str:
        """Determine element type for LOD configuration"""
        element_name = element.__class__.__name__.lower()

        if 'label' in element_name or 'text' in element_name:
            return 'labels'
        elif 'indicator' in element_name:
            return 'indicators'
        else:
            return 'blocks'

    def get_statistics(self) -> Dict[str, Any]:
        """Get LOD statistics"""
        return {
            'current_lod': self.current_lod.value,
            **self.lod_stats
        }


class RenderCache:
    """
    Render cache for expensive visual elements.

    Caches rendered primitives and visual data to avoid
    redundant computations across frames.
    """

    def __init__(self, settings: PerformanceSettings):
        """Initialize render cache"""
        self.settings = settings

        # Cache storage
        self.geometry_cache: Dict[str, Tuple[List[DrawingPrimitive], float]] = {}  # (primitives, timestamp)
        self.texture_cache: Dict[str, Tuple[Any, float]] = {}  # (texture_data, timestamp)

        # Cache statistics
        self.cache_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_evictions': 0,
            'memory_usage_mb': 0.0
        }

        # Cleanup tracking
        self.last_cleanup_time = time.time()

    def get_cached_geometry(self, cache_key: str) -> Optional[List[DrawingPrimitive]]:
        """Get cached geometry primitives"""
        if not self.settings.enable_caching:
            return None

        if cache_key in self.geometry_cache:
            primitives, timestamp = self.geometry_cache[cache_key]

            # Check if cache entry is still valid
            if time.time() - timestamp <= self.settings.cache_ttl_seconds:
                self.cache_stats['cache_hits'] += 1
                return primitives
            else:
                # Remove expired entry
                del self.geometry_cache[cache_key]

        self.cache_stats['cache_misses'] += 1
        return None

    def cache_geometry(self, cache_key: str, primitives: List[DrawingPrimitive]) -> None:
        """Cache geometry primitives"""
        if not self.settings.enable_caching:
            return

        # Check cache size limits
        if len(self.geometry_cache) >= self.settings.max_cache_entries:
            self._evict_oldest_entries()

        self.geometry_cache[cache_key] = (primitives, time.time())
        self._update_memory_usage()

    def get_cached_texture(self, cache_key: str) -> Optional[Any]:
        """Get cached texture data"""
        if not self.settings.enable_caching:
            return None

        if cache_key in self.texture_cache:
            texture_data, timestamp = self.texture_cache[cache_key]

            if time.time() - timestamp <= self.settings.cache_ttl_seconds:
                self.cache_stats['cache_hits'] += 1
                return texture_data
            else:
                del self.texture_cache[cache_key]

        self.cache_stats['cache_misses'] += 1
        return None

    def cache_texture(self, cache_key: str, texture_data: Any) -> None:
        """Cache texture data"""
        if not self.settings.enable_caching:
            return

        if len(self.texture_cache) >= self.settings.max_cache_entries:
            self._evict_oldest_entries()

        self.texture_cache[cache_key] = (texture_data, time.time())
        self._update_memory_usage()

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.geometry_cache.clear()
        self.texture_cache.clear()
        self.cache_stats['memory_usage_mb'] = 0.0

    def cleanup_expired(self) -> None:
        """Clean up expired cache entries"""
        current_time = time.time()
        ttl = self.settings.cache_ttl_seconds

        # Clean geometry cache
        expired_geo_keys = []
        for key, (_, timestamp) in self.geometry_cache.items():
            if current_time - timestamp > ttl:
                expired_geo_keys.append(key)

        for key in expired_geo_keys:
            del self.geometry_cache[key]
            self.cache_stats['cache_evictions'] += 1

        # Clean texture cache
        expired_tex_keys = []
        for key, (_, timestamp) in self.texture_cache.items():
            if current_time - timestamp > ttl:
                expired_tex_keys.append(key)

        for key in expired_tex_keys:
            del self.texture_cache[key]
            self.cache_stats['cache_evictions'] += 1

        self._update_memory_usage()
        self.last_cleanup_time = current_time

    def _evict_oldest_entries(self) -> None:
        """Evict oldest cache entries when at capacity"""
        # Evict 25% of entries
        eviction_count = max(1, len(self.geometry_cache) // 4)

        # Sort by timestamp (oldest first)
        sorted_geo = sorted(self.geometry_cache.items(), key=lambda x: x[1][1])
        for i in range(min(eviction_count, len(sorted_geo))):
            key = sorted_geo[i][0]
            del self.geometry_cache[key]
            self.cache_stats['cache_evictions'] += 1

        # Same for texture cache
        sorted_tex = sorted(self.texture_cache.items(), key=lambda x: x[1][1])
        for i in range(min(eviction_count, len(sorted_tex))):
            key = sorted_tex[i][0]
            del self.texture_cache[key]
            self.cache_stats['cache_evictions'] += 1

    def _update_memory_usage(self) -> None:
        """Update estimated memory usage"""
        # Rough estimation (would be more accurate with actual measurements)
        geo_entries = len(self.geometry_cache)
        tex_entries = len(self.texture_cache)

        # Estimate ~1KB per geometry entry, ~10KB per texture entry
        estimated_mb = (geo_entries * 0.001) + (tex_entries * 0.01)
        self.cache_stats['memory_usage_mb'] = estimated_mb

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = 0.0
        total_requests = self.cache_stats['cache_hits'] + self.cache_stats['cache_misses']
        if total_requests > 0:
            hit_rate = self.cache_stats['cache_hits'] / total_requests

        return {
            **self.cache_stats,
            'hit_rate': hit_rate,
            'total_entries': len(self.geometry_cache) + len(self.texture_cache),
            'geometry_entries': len(self.geometry_cache),
            'texture_entries': len(self.texture_cache)
        }


class BatchRenderer:
    """
    Batch renderer for efficient primitive rendering.

    Groups similar primitives together to reduce draw calls
    and improve rendering performance.
    """

    def __init__(self, settings: PerformanceSettings):
        """Initialize batch renderer"""
        self.settings = settings

        # Batch storage
        self.primitive_batches: Dict[str, List[DrawingPrimitive]] = {}

        # Batch statistics
        self.batch_stats = {
            'batches_created': 0,
            'primitives_batched': 0,
            'draw_calls_saved': 0,
            'batch_efficiency': 0.0
        }

    def add_primitive(self, primitive: DrawingPrimitive) -> None:
        """Add primitive to appropriate batch"""
        if not self.settings.enable_batching:
            return

        batch_key = self._get_batch_key(primitive)

        if batch_key not in self.primitive_batches:
            self.primitive_batches[batch_key] = []
            self.batch_stats['batches_created'] += 1

        batch = self.primitive_batches[batch_key]

        # Check batch size limits
        if len(batch) < self.settings.max_batch_size:
            batch.append(primitive)
            self.batch_stats['primitives_batched'] += 1
        else:
            # Create new batch
            new_batch_key = f"{batch_key}_{len(batch) // self.settings.max_batch_size}"
            self.primitive_batches[new_batch_key] = [primitive]
            self.batch_stats['batches_created'] += 1
            self.batch_stats['primitives_batched'] += 1

    def get_batches(self) -> List[Tuple[str, List[DrawingPrimitive]]]:
        """Get all batches for rendering"""
        batches = []

        for batch_key, primitives in self.primitive_batches.items():
            if primitives:  # Only return non-empty batches
                batches.append((batch_key, primitives))

        # Calculate efficiency
        total_primitives = sum(len(batch[1]) for batch in batches)
        if total_primitives > 0:
            self.batch_stats['batch_efficiency'] = total_primitives / max(1, len(batches))
            self.batch_stats['draw_calls_saved'] = max(0, total_primitives - len(batches))

        return batches

    def clear_batches(self) -> None:
        """Clear all primitive batches"""
        self.primitive_batches.clear()

    def _get_batch_key(self, primitive: DrawingPrimitive) -> str:
        """Generate batch key for primitive grouping"""
        if not self.settings.batch_by_material:
            return primitive.primitive_type.value

        # Create more specific batch key including material properties
        key_parts = [
            primitive.primitive_type.value,
            str(primitive.fill_color) if primitive.fill_color else "no_fill",
            str(primitive.stroke_color) if primitive.stroke_color else "no_stroke",
            str(int(primitive.stroke_width)) if primitive.stroke_width else "0"
        ]

        return "_".join(key_parts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get batch statistics"""
        return self.batch_stats.copy()


class PerformanceMonitor:
    """
    Performance monitoring and adaptive quality system.

    Monitors rendering performance and automatically adjusts
    quality settings to maintain target frame rate.
    """

    def __init__(self, settings: PerformanceSettings):
        """Initialize performance monitor"""
        self.settings = settings

        # Performance tracking
        self.frame_times: List[float] = []
        self.max_frame_history = 60  # Track last 60 frames

        # Adaptive quality state
        self.current_quality_factor = 1.0
        self.last_quality_adjustment = time.time()
        self.quality_adjustment_cooldown = 1.0  # 1 second

        # Statistics
        self.performance_stats = RenderStats()

    def begin_frame(self) -> None:
        """Begin frame performance measurement"""
        self.frame_start_time = time.perf_counter()
        self.performance_stats.reset()

    def end_frame(self) -> None:
        """End frame and update performance statistics"""
        frame_time = (time.perf_counter() - self.frame_start_time) * 1000

        # Update frame time history
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)

        # Update stats
        self.performance_stats.frame_time_ms = frame_time

        # Adaptive quality adjustment
        if self.settings.adaptive_quality:
            self._adjust_quality()

    def get_average_fps(self) -> float:
        """Get average FPS over recent frames"""
        if not self.frame_times:
            return 0.0

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1000.0 / max(0.1, avg_frame_time)

    def get_performance_factor(self) -> float:
        """Get current performance factor (0.0 to 1.0)"""
        target_frame_time = self.settings.min_frame_time_ms
        current_fps = self.get_average_fps()
        target_fps = self.settings.target_fps

        if current_fps >= target_fps:
            return 1.0
        else:
            return max(0.1, current_fps / target_fps)

    def _adjust_quality(self) -> None:
        """Automatically adjust quality based on performance"""
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_quality_adjustment < self.quality_adjustment_cooldown:
            return

        performance_factor = self.get_performance_factor()

        # Adjust quality factor
        if performance_factor < 0.8:  # Performance is poor
            self.current_quality_factor = max(0.3, self.current_quality_factor * 0.9)
            self.last_quality_adjustment = current_time
        elif performance_factor > 0.95:  # Performance is good
            self.current_quality_factor = min(1.0, self.current_quality_factor * 1.05)
            self.last_quality_adjustment = current_time

        self.performance_stats.quality_factor = self.current_quality_factor

    def get_statistics(self) -> RenderStats:
        """Get current performance statistics"""
        self.performance_stats.quality_factor = self.current_quality_factor
        return self.performance_stats

    def should_skip_frame(self) -> bool:
        """Determine if frame should be skipped for performance"""
        if not self.frame_times:
            return False

        # Skip if consistently over target frame time
        recent_frames = self.frame_times[-5:] if len(self.frame_times) >= 5 else self.frame_times
        avg_recent = sum(recent_frames) / len(recent_frames)

        return avg_recent > self.settings.min_frame_time_ms * 1.5