"""
Mathematical utilities for Neural-Klotski system.

Provides elementary mathematical operations including:
- 2D vector operations for position and velocity
- Spatial distance calculations along lag axis
- Bounds checking functions
- Numerical integration utilities (Euler method)

All functions are designed for precision and mathematical correctness
according to the specification document.
"""

import numpy as np
from typing import Tuple, Union
import math


# Type aliases for clarity
Position2D = Tuple[float, float]  # (activation, lag)
Vector2D = Tuple[float, float]    # (velocity_activation, velocity_lag)


class Vector2D:
    """2D vector class for position and velocity operations"""

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> 'Vector2D':
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> 'Vector2D':
        if scalar == 0:
            raise ValueError("Cannot divide vector by zero")
        return Vector2D(self.x / scalar, self.y / scalar)

    def magnitude(self) -> float:
        """Calculate vector magnitude (Euclidean norm)"""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self) -> 'Vector2D':
        """Return normalized vector (unit vector in same direction)"""
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return self / mag

    def dot(self, other: 'Vector2D') -> float:
        """Calculate dot product with another vector"""
        return self.x * other.x + self.y * other.y

    def distance_to(self, other: 'Vector2D') -> float:
        """Calculate Euclidean distance to another vector"""
        return (self - other).magnitude()

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple (x, y)"""
        return (self.x, self.y)

    def __repr__(self) -> str:
        return f"Vector2D({self.x:.3f}, {self.y:.3f})"

    def __eq__(self, other: 'Vector2D') -> bool:
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9


def lag_distance(lag1: float, lag2: float) -> float:
    """
    Calculate distance between two positions on the lag axis.

    Args:
        lag1: First lag position
        lag2: Second lag position

    Returns:
        Absolute distance |lag2 - lag1|
    """
    return abs(lag2 - lag1)


def signal_propagation_delay(source_lag: float, target_lag: float, signal_speed: float) -> float:
    """
    Calculate signal propagation delay based on lag positions.

    From Section 5.4.2: delay = lag_distance / v_signal

    Args:
        source_lag: Source block lag position
        target_lag: Target block lag position
        signal_speed: Signal propagation speed (lag units per timestep)

    Returns:
        Propagation delay in timesteps
    """
    if signal_speed <= 0:
        raise ValueError("Signal speed must be positive")

    distance = lag_distance(source_lag, target_lag)
    return distance / signal_speed


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value to be within [min_val, max_val] range.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
    if min_val > max_val:
        raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")

    return max(min_val, min(value, max_val))


def is_in_bounds(value: float, min_val: float, max_val: float) -> bool:
    """
    Check if value is within specified bounds.

    Args:
        value: Value to check
        min_val: Minimum bound (inclusive)
        max_val: Maximum bound (inclusive)

    Returns:
        True if min_val <= value <= max_val
    """
    return min_val <= value <= max_val


def euler_integration_step(position: float, velocity: float, acceleration: float, dt: float) -> Tuple[float, float]:
    """
    Perform one step of Euler integration for 1D motion.

    From Section 9.2.1:
    - x(t + dt) = x(t) + v(t) * dt
    - v(t + dt) = v(t) * damping + acceleration * dt

    Args:
        position: Current position
        velocity: Current velocity
        acceleration: Current acceleration (force/mass)
        dt: Time step size

    Returns:
        Tuple of (new_position, new_velocity)
    """
    new_position = position + velocity * dt
    new_velocity = velocity + acceleration * dt

    return new_position, new_velocity


def euler_integration_step_damped(position: float, velocity: float, force: float,
                                 mass: float, damping: float, dt: float) -> Tuple[float, float]:
    """
    Perform one step of Euler integration with damping.

    From Section 9.2.1:
    - x(t + dt) = x(t) + v(t) * dt
    - v(t + dt) = v(t) * (1 - γ*dt) + (F_total/m) * dt

    Args:
        position: Current position
        velocity: Current velocity
        force: Total applied force
        mass: Block mass
        damping: Damping coefficient γ
        dt: Time step size

    Returns:
        Tuple of (new_position, new_velocity)
    """
    if mass <= 0:
        raise ValueError("Mass must be positive")
    if damping < 0:
        raise ValueError("Damping must be non-negative")

    new_position = position + velocity * dt
    damping_factor = 1.0 - damping * dt
    acceleration = force / mass
    new_velocity = velocity * damping_factor + acceleration * dt

    return new_position, new_velocity


def spring_force(position: float, rest_position: float = 0.0, spring_constant: float = 1.0) -> float:
    """
    Calculate spring restoring force.

    From Section 9.2.1: F_spring = -k * (x - x_rest)

    Args:
        position: Current position
        rest_position: Rest position (typically 0)
        spring_constant: Spring constant k

    Returns:
        Spring force (negative pulls toward rest position)
    """
    if spring_constant < 0:
        raise ValueError("Spring constant must be non-negative")

    return -spring_constant * (position - rest_position)


def exponential_decay(initial_value: float, time: float, time_constant: float) -> float:
    """
    Calculate exponential decay.

    From Section 9.3: C(t) = C(0) * exp(-t/τ)

    Args:
        initial_value: Initial value C(0)
        time: Time elapsed
        time_constant: Decay time constant τ

    Returns:
        Decayed value
    """
    if time_constant <= 0:
        raise ValueError("Time constant must be positive")
    if time < 0:
        raise ValueError("Time must be non-negative")

    return initial_value * math.exp(-time / time_constant)


def discrete_laplacian_2d(concentration_grid: np.ndarray, i: int, j: int, dx: float = 1.0) -> float:
    """
    Calculate discrete 2D Laplacian at grid point (i, j).

    For diffusion equation: ∂C/∂t = D * ∇²C
    ∇²C ≈ [C(i+1,j) + C(i-1,j) + C(i,j+1) + C(i,j-1) - 4*C(i,j)] / dx²

    Args:
        concentration_grid: 2D array of concentration values
        i: Row index
        j: Column index
        dx: Spatial grid spacing

    Returns:
        Discrete Laplacian value
    """
    rows, cols = concentration_grid.shape

    # Handle boundary conditions (zero gradient at boundaries)
    c_center = concentration_grid[i, j]
    c_left = concentration_grid[i, j-1] if j > 0 else c_center
    c_right = concentration_grid[i, j+1] if j < cols-1 else c_center
    c_up = concentration_grid[i-1, j] if i > 0 else c_center
    c_down = concentration_grid[i+1, j] if i < rows-1 else c_center

    laplacian = (c_left + c_right + c_up + c_down - 4 * c_center) / (dx * dx)

    return laplacian


def diffusion_step(concentration_grid: np.ndarray, diffusion_coeff: float,
                  dt: float, dx: float = 1.0) -> np.ndarray:
    """
    Perform one diffusion time step on entire 2D grid.

    Args:
        concentration_grid: 2D concentration field
        diffusion_coeff: Diffusion coefficient D
        dt: Time step
        dx: Spatial grid spacing

    Returns:
        Updated concentration grid
    """
    rows, cols = concentration_grid.shape
    new_grid = concentration_grid.copy()

    for i in range(rows):
        for j in range(cols):
            laplacian = discrete_laplacian_2d(concentration_grid, i, j, dx)
            new_grid[i, j] += diffusion_coeff * laplacian * dt

    # Ensure non-negative concentrations
    new_grid = np.maximum(new_grid, 0.0)

    return new_grid


def threshold_crossing_from_below(prev_position: float, curr_position: float,
                                threshold: float) -> bool:
    """
    Detect threshold crossing from below.

    Critical requirement from Section 9.2.2: Block must cross threshold from below
    to trigger firing.

    Args:
        prev_position: Position at previous timestep
        curr_position: Position at current timestep
        threshold: Threshold value

    Returns:
        True if threshold was crossed from below
    """
    return prev_position < threshold and curr_position >= threshold


def calculate_k_nearest_neighbors(target_lag: float, all_lags: list, k: int) -> list:
    """
    Find k nearest neighbors by lag distance.

    Args:
        target_lag: Lag position of target block
        all_lags: List of all lag positions
        k: Number of neighbors to find

    Returns:
        List of k nearest lag positions (excluding target itself)
    """
    # Calculate distances to all other positions
    distances = []
    for lag in all_lags:
        if lag != target_lag:  # Exclude self
            dist = lag_distance(target_lag, lag)
            distances.append((dist, lag))

    # Sort by distance and return k nearest
    distances.sort(key=lambda x: x[0])
    return [lag for _, lag in distances[:k]]


def wire_effective_strength(base_strength: float, dye_concentration: float,
                          enhancement_factor: float = 1.0) -> float:
    """
    Calculate effective wire strength with dye enhancement.

    From Section 3.4.3: effective_strength = base_strength * (1 + α * C_local)

    Args:
        base_strength: Base wire strength
        dye_concentration: Local dye concentration
        enhancement_factor: Enhancement factor α

    Returns:
        Effective strength
    """
    if dye_concentration < 0:
        raise ValueError("Dye concentration must be non-negative")

    return base_strength * (1.0 + enhancement_factor * dye_concentration)


if __name__ == "__main__":
    # Test basic vector operations
    v1 = Vector2D(3.0, 4.0)
    v2 = Vector2D(1.0, 2.0)

    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1.magnitude() = {v1.magnitude()}")
    print(f"v1.distance_to(v2) = {v1.distance_to(v2)}")

    # Test signal propagation
    delay = signal_propagation_delay(50.0, 100.0, 100.0)
    print(f"Signal delay from lag 50 to lag 100 at speed 100: {delay} timesteps")

    # Test threshold crossing
    crossed = threshold_crossing_from_below(49.5, 50.5, 50.0)
    print(f"Threshold crossing from 49.5 to 50.5 (threshold=50): {crossed}")

    # Test integration
    pos, vel = euler_integration_step_damped(0.0, 5.0, 10.0, 1.0, 0.1, 0.1)
    print(f"Integration step: new_pos={pos:.3f}, new_vel={vel:.3f}")

    print("All math_utils tests completed successfully!")