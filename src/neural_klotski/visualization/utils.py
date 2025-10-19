"""
Visualization Utilities

Shared utilities and helper functions for the Neural-Klotski visualization system.
Provides common functionality for color management, coordinate transformations,
animation calculations, and performance monitoring.
"""

import math
import time
import psutil
from typing import Tuple, List, Optional, Any
import numpy as np
from dataclasses import dataclass


class ColorUtils:
    """Utilities for color management and conversion"""

    @staticmethod
    def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
        """Convert RGB tuple (0.0-1.0) to hex string"""
        r, g, b = [int(c * 255) for c in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
        """Convert hex string to RGB tuple (0.0-1.0)"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    @staticmethod
    def hsv_to_rgb(hsv: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Convert HSV to RGB"""
        h, s, v = hsv
        if s == 0.0:
            return (v, v, v)

        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        idx = i % 6
        if idx == 0:
            return (v, t, p)
        elif idx == 1:
            return (q, v, p)
        elif idx == 2:
            return (p, v, t)
        elif idx == 3:
            return (p, q, v)
        elif idx == 4:
            return (t, p, v)
        else:
            return (v, p, q)

    @staticmethod
    def rgb_to_hsv(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Convert RGB to HSV"""
        r, g, b = rgb
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val

        # Value
        v = max_val

        # Saturation
        s = 0 if max_val == 0 else diff / max_val

        # Hue
        if diff == 0:
            h = 0
        elif max_val == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360

        return (h / 360.0, s, v)

    @staticmethod
    def interpolate_rgb(color1: Tuple[float, float, float],
                       color2: Tuple[float, float, float],
                       t: float) -> Tuple[float, float, float]:
        """Interpolate between two RGB colors"""
        t = max(0.0, min(1.0, t))  # Clamp t to [0, 1]
        return (
            color1[0] + t * (color2[0] - color1[0]),
            color1[1] + t * (color2[1] - color1[1]),
            color1[2] + t * (color2[2] - color1[2])
        )

    @staticmethod
    def interpolate_hsv(color1: Tuple[float, float, float],
                       color2: Tuple[float, float, float],
                       t: float) -> Tuple[float, float, float]:
        """Interpolate between two colors in HSV space"""
        hsv1 = ColorUtils.rgb_to_hsv(color1)
        hsv2 = ColorUtils.rgb_to_hsv(color2)

        # Handle hue wraparound
        h1, s1, v1 = hsv1
        h2, s2, v2 = hsv2

        if abs(h2 - h1) > 0.5:
            if h1 > h2:
                h2 += 1.0
            else:
                h1 += 1.0

        h_interp = (h1 + t * (h2 - h1)) % 1.0
        s_interp = s1 + t * (s2 - s1)
        v_interp = v1 + t * (v2 - v1)

        return ColorUtils.hsv_to_rgb((h_interp, s_interp, v_interp))

    @staticmethod
    def add_alpha(rgb: Tuple[float, float, float], alpha: float) -> Tuple[float, float, float, float]:
        """Add alpha channel to RGB color"""
        return rgb + (alpha,)

    @staticmethod
    def darken(rgb: Tuple[float, float, float], factor: float) -> Tuple[float, float, float]:
        """Darken color by factor (0.0 = black, 1.0 = original)"""
        return tuple(c * factor for c in rgb)

    @staticmethod
    def lighten(rgb: Tuple[float, float, float], factor: float) -> Tuple[float, float, float]:
        """Lighten color by factor (1.0 = original, 2.0 = double brightness)"""
        return tuple(min(1.0, c * factor) for c in rgb)


class CoordinateUtils:
    """Utilities for coordinate system transformations"""

    @staticmethod
    def world_to_screen(world_pos: Tuple[float, float],
                       viewport_center: Tuple[float, float],
                       viewport_size: Tuple[float, float],
                       screen_size: Tuple[int, int],
                       zoom: float = 1.0) -> Tuple[int, int]:
        """Convert world coordinates to screen pixels"""
        world_x, world_y = world_pos
        center_x, center_y = viewport_center
        view_w, view_h = viewport_size
        screen_w, screen_h = screen_size

        # Transform to normalized viewport coordinates (-1 to 1)
        norm_x = (world_x - center_x + view_w/2) / (view_w/2) / zoom
        norm_y = (world_y - center_y + view_h/2) / (view_h/2) / zoom

        # Transform to screen coordinates
        screen_x = int((norm_x + 1) * screen_w / 2)
        screen_y = int((1 - norm_y) * screen_h / 2)  # Flip Y axis

        return (screen_x, screen_y)

    @staticmethod
    def screen_to_world(screen_pos: Tuple[int, int],
                       viewport_center: Tuple[float, float],
                       viewport_size: Tuple[float, float],
                       screen_size: Tuple[int, int],
                       zoom: float = 1.0) -> Tuple[float, float]:
        """Convert screen pixels to world coordinates"""
        screen_x, screen_y = screen_pos
        center_x, center_y = viewport_center
        view_w, view_h = viewport_size
        screen_w, screen_h = screen_size

        # Transform to normalized coordinates (-1 to 1)
        norm_x = (2 * screen_x / screen_w) - 1
        norm_y = 1 - (2 * screen_y / screen_h)  # Flip Y axis

        # Transform to world coordinates
        world_x = center_x - view_w/2 + norm_x * (view_w/2) * zoom
        world_y = center_y - view_h/2 + norm_y * (view_h/2) * zoom

        return (world_x, world_y)

    @staticmethod
    def distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def angle_between(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate angle in radians from pos1 to pos2"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return math.atan2(dy, dx)

    @staticmethod
    def rotate_point(point: Tuple[float, float], center: Tuple[float, float], angle: float) -> Tuple[float, float]:
        """Rotate point around center by angle (radians)"""
        px, py = point
        cx, cy = center

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Translate to origin
        px -= cx
        py -= cy

        # Rotate
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a

        # Translate back
        return (rx + cx, ry + cy)

    @staticmethod
    def clamp_to_bounds(pos: Tuple[float, float],
                       bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Clamp position to rectangular bounds (min_x, min_y, max_x, max_y)"""
        x, y = pos
        min_x, min_y, max_x, max_y = bounds
        return (max(min_x, min(max_x, x)), max(min_y, min(max_y, y)))


class AnimationUtils:
    """Utilities for animation calculations and interpolation"""

    @staticmethod
    def linear_interpolate(start: float, end: float, t: float) -> float:
        """Linear interpolation between start and end"""
        return start + t * (end - start)

    @staticmethod
    def ease_in_out(t: float) -> float:
        """Ease-in-out curve (smooth acceleration/deceleration)"""
        if t < 0.5:
            return 2 * t * t
        else:
            return 1 - 2 * (1 - t) * (1 - t)

    @staticmethod
    def ease_in(t: float) -> float:
        """Ease-in curve (smooth acceleration)"""
        return t * t

    @staticmethod
    def ease_out(t: float) -> float:
        """Ease-out curve (smooth deceleration)"""
        return 1 - (1 - t) * (1 - t)

    @staticmethod
    def cubic_bezier(t: float, p0: float, p1: float, p2: float, p3: float) -> float:
        """Cubic Bezier curve interpolation"""
        u = 1 - t
        return (u*u*u * p0 +
                3*u*u*t * p1 +
                3*u*t*t * p2 +
                t*t*t * p3)

    @staticmethod
    def spring_interpolate(current: float, target: float, velocity: float,
                          stiffness: float, damping: float, dt: float) -> Tuple[float, float]:
        """Spring-based interpolation returning (new_position, new_velocity)"""
        force = (target - current) * stiffness
        velocity += force * dt
        velocity *= (1 - damping * dt)
        current += velocity * dt
        return (current, velocity)

    @staticmethod
    def smooth_damp(current: float, target: float, velocity: float,
                   smooth_time: float, dt: float) -> Tuple[float, float]:
        """Smooth damping interpolation (Unity-style SmoothDamp)"""
        smooth_time = max(0.0001, smooth_time)
        omega = 2.0 / smooth_time

        x = omega * dt
        exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)

        change = current - target
        original_to = target

        max_change = float('inf')  # No max speed limit for now
        change = max(-max_change, min(max_change, change))
        target = current - change

        temp = (velocity + omega * change) * dt
        velocity = (velocity - omega * temp) * exp
        output = target + (change + temp) * exp

        # Prevent overshooting
        if (original_to - current > 0.0) == (output > original_to):
            output = original_to
            velocity = (output - original_to) / dt

        return (output, velocity)

    @staticmethod
    def calculate_frame_time(target_fps: float) -> float:
        """Calculate target frame time from FPS"""
        return 1.0 / max(1.0, target_fps)

    @staticmethod
    def lerp_angle(a1: float, a2: float, t: float) -> float:
        """Interpolate between two angles (radians) taking shortest path"""
        diff = a2 - a1
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return a1 + diff * t


@dataclass
class PerformanceStats:
    """Performance statistics container"""
    fps: float = 0.0
    frame_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    render_time_ms: float = 0.0
    update_time_ms: float = 0.0


class PerformanceUtils:
    """Utilities for performance monitoring and optimization"""

    def __init__(self):
        self.frame_times: List[float] = []
        self.last_fps_update = 0.0
        self.fps_update_interval = 1.0  # Update FPS every second
        self.process = psutil.Process()

    def start_frame(self) -> float:
        """Start frame timing, return current time"""
        return time.perf_counter()

    def end_frame(self, start_time: float) -> float:
        """End frame timing, return frame duration"""
        frame_time = time.perf_counter() - start_time
        self.frame_times.append(frame_time)

        # Keep only recent frame times (for rolling average)
        if len(self.frame_times) > 120:  # 2 seconds at 60fps
            self.frame_times = self.frame_times[-120:]

        return frame_time

    def get_fps(self) -> float:
        """Get current FPS based on recent frame times"""
        if not self.frame_times:
            return 0.0

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def get_frame_time_ms(self) -> float:
        """Get average frame time in milliseconds"""
        if not self.frame_times:
            return 0.0

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return avg_frame_time * 1000.0

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def get_cpu_percent(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return self.process.cpu_percent()
        except:
            return 0.0

    def get_performance_stats(self) -> PerformanceStats:
        """Get comprehensive performance statistics"""
        return PerformanceStats(
            fps=self.get_fps(),
            frame_time_ms=self.get_frame_time_ms(),
            memory_usage_mb=self.get_memory_usage_mb(),
            cpu_percent=self.get_cpu_percent()
        )

    def should_skip_frame(self, target_fps: float, skip_threshold: float = 0.5) -> bool:
        """Determine if frame should be skipped for performance"""
        current_fps = self.get_fps()
        return current_fps < target_fps * skip_threshold

    def adaptive_quality_factor(self, target_fps: float) -> float:
        """Get quality factor (0.0-1.0) based on performance"""
        current_fps = self.get_fps()
        if current_fps >= target_fps:
            return 1.0
        elif current_fps <= target_fps * 0.5:
            return 0.5
        else:
            return current_fps / target_fps

    @staticmethod
    def format_bytes(bytes_value: float) -> str:
        """Format bytes in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"

    @staticmethod
    def format_time_ms(time_seconds: float) -> str:
        """Format time in milliseconds with appropriate precision"""
        ms = time_seconds * 1000
        if ms < 1:
            return f"{ms:.2f} ms"
        elif ms < 10:
            return f"{ms:.1f} ms"
        else:
            return f"{ms:.0f} ms"


# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceUtils:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceUtils()
    return _performance_monitor


# Utility functions for common tasks
def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))


def map_range(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Map value from input range to output range"""
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)


def smooth_step(edge0: float, edge1: float, x: float) -> float:
    """Hermite interpolation (smooth step function)"""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def normalize_vector(vector: Tuple[float, float]) -> Tuple[float, float]:
    """Normalize 2D vector to unit length"""
    x, y = vector
    length = math.sqrt(x*x + y*y)
    if length == 0:
        return (0.0, 0.0)
    return (x/length, y/length)


def vector_add(v1: Tuple[float, float], v2: Tuple[float, float]) -> Tuple[float, float]:
    """Add two 2D vectors"""
    return (v1[0] + v2[0], v1[1] + v2[1])


def vector_scale(vector: Tuple[float, float], scale: float) -> Tuple[float, float]:
    """Scale 2D vector by scalar"""
    return (vector[0] * scale, vector[1] * scale)


def vector_dot(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Calculate dot product of two 2D vectors"""
    return v1[0] * v2[0] + v1[1] * v2[1]