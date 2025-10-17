"""
Dye field system for Neural-Klotski learning signals.

Implements 2D spatial dye fields with diffusion and decay according to Section 9.3
of the specification. Handles red, blue, and yellow dye concentrations that enhance
wire strengths and provide learning signals for plasticity mechanisms.
"""

from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.core.block import BlockColor
from neural_klotski.math_utils import (
    discrete_laplacian_2d, diffusion_step, exponential_decay, clamp
)
from neural_klotski.config import DyeConfig


class DyeColor(Enum):
    """Dye color types matching wire colors"""
    RED = "red"
    BLUE = "blue"
    YELLOW = "yellow"

    @classmethod
    def from_block_color(cls, block_color: BlockColor) -> 'DyeColor':
        """Convert BlockColor to DyeColor"""
        if block_color == BlockColor.RED:
            return cls.RED
        elif block_color == BlockColor.BLUE:
            return cls.BLUE
        elif block_color == BlockColor.YELLOW:
            return cls.YELLOW
        else:
            raise ValueError(f"Unknown block color: {block_color}")


@dataclass
class SpatialBounds:
    """Defines spatial bounds for dye field grid"""
    activation_min: float
    activation_max: float
    lag_min: float
    lag_max: float
    grid_resolution: float = 1.0  # Spatial resolution (units per grid cell)

    def __post_init__(self):
        """Validate spatial bounds"""
        if self.activation_max <= self.activation_min:
            raise ValueError("activation_max must be > activation_min")
        if self.lag_max <= self.lag_min:
            raise ValueError("lag_max must be > lag_min")
        if self.grid_resolution <= 0:
            raise ValueError("grid_resolution must be positive")

    def get_grid_shape(self) -> Tuple[int, int]:
        """Get grid dimensions (rows, cols) for spatial bounds"""
        rows = int(np.ceil((self.lag_max - self.lag_min) / self.grid_resolution))
        cols = int(np.ceil((self.activation_max - self.activation_min) / self.grid_resolution))
        return max(1, rows), max(1, cols)

    def spatial_to_grid(self, activation: float, lag: float) -> Tuple[int, int]:
        """Convert spatial coordinates to grid indices"""
        # Clamp coordinates to bounds
        activation = clamp(activation, self.activation_min, self.activation_max - self.grid_resolution)
        lag = clamp(lag, self.lag_min, self.lag_max - self.grid_resolution)

        # Convert to grid indices
        col = int((activation - self.activation_min) / self.grid_resolution)
        row = int((lag - self.lag_min) / self.grid_resolution)

        # Ensure within grid bounds
        rows, cols = self.get_grid_shape()
        row = clamp(row, 0, rows - 1)
        col = clamp(col, 0, cols - 1)

        return int(row), int(col)

    def grid_to_spatial(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid indices to spatial coordinates (center of cell)"""
        activation = self.activation_min + (col + 0.5) * self.grid_resolution
        lag = self.lag_min + (row + 0.5) * self.grid_resolution
        return activation, lag


class DyeField:
    """
    2D spatial dye field with diffusion and decay dynamics.

    Manages a single color of dye concentration across the 2D spatial grid
    with proper diffusion and decay mechanics according to Section 9.3.
    """

    def __init__(self, dye_color: DyeColor, spatial_bounds: SpatialBounds):
        """
        Initialize dye field.

        Args:
            dye_color: Color of this dye field
            spatial_bounds: Spatial bounds and grid resolution
        """
        self.dye_color = dye_color
        self.spatial_bounds = spatial_bounds

        # Initialize concentration grid
        rows, cols = spatial_bounds.get_grid_shape()
        self.concentration = np.zeros((rows, cols), dtype=np.float64)

        # Track total dye for conservation monitoring
        self.total_dye_injected = 0.0
        self.total_dye_decayed = 0.0

    def get_concentration_at(self, activation: float, lag: float) -> float:
        """
        Get dye concentration at spatial coordinates.

        Args:
            activation: Activation axis coordinate
            lag: Lag axis coordinate

        Returns:
            Dye concentration at the specified location
        """
        row, col = self.spatial_bounds.spatial_to_grid(activation, lag)
        return float(self.concentration[row, col])

    def inject_dye(self, activation: float, lag: float, amount: float,
                   spread_radius: float = 1.0):
        """
        Inject dye at specified spatial location.

        Args:
            activation: Activation axis coordinate for injection
            lag: Lag axis coordinate for injection
            amount: Amount of dye to inject
            spread_radius: Radius for dye spread (in grid units)
        """
        if amount <= 0:
            return

        center_row, center_col = self.spatial_bounds.spatial_to_grid(activation, lag)
        rows, cols = self.concentration.shape

        # Calculate spread pattern (Gaussian-like)
        spread_radius_grid = spread_radius / self.spatial_bounds.grid_resolution

        # Determine affected region
        radius_int = max(1, int(np.ceil(spread_radius_grid)))

        total_weight = 0.0
        injection_pattern = []

        for dr in range(-radius_int, radius_int + 1):
            for dc in range(-radius_int, radius_int + 1):
                r = center_row + dr
                c = center_col + dc

                if 0 <= r < rows and 0 <= c < cols:
                    # Calculate distance from center
                    distance = np.sqrt(dr**2 + dc**2)

                    if distance <= spread_radius_grid:
                        # Gaussian-like weight
                        weight = np.exp(-(distance**2) / (2 * (spread_radius_grid/2)**2))
                        injection_pattern.append((r, c, weight))
                        total_weight += weight

        # Normalize and apply injection
        if total_weight > 0:
            for r, c, weight in injection_pattern:
                injection_amount = amount * (weight / total_weight)
                self.concentration[r, c] += injection_amount

        self.total_dye_injected += amount

    def step_diffusion(self, config: DyeConfig, dt: float):
        """
        Perform one diffusion step.

        Args:
            config: Dye configuration parameters
            dt: Time step size
        """
        # Apply diffusion using existing math utilities
        self.concentration = diffusion_step(
            self.concentration,
            config.diffusion_coefficient,
            dt,
            self.spatial_bounds.grid_resolution
        )

    def step_decay(self, config: DyeConfig, dt: float):
        """
        Perform exponential decay step.

        Args:
            config: Dye configuration parameters
            dt: Time step size
        """
        # Calculate decay factor for this timestep
        decay_factor = np.exp(-dt / config.decay_time_constant)

        # Track total decay
        pre_decay_total = np.sum(self.concentration)

        # Apply decay
        self.concentration *= decay_factor

        # Track decay amount
        post_decay_total = np.sum(self.concentration)
        self.total_dye_decayed += (pre_decay_total - post_decay_total)

    def step_dynamics(self, config: DyeConfig, dt: float):
        """
        Perform complete dye dynamics step (diffusion + decay).

        Args:
            config: Dye configuration parameters
            dt: Time step size
        """
        self.step_diffusion(config, dt)
        self.step_decay(config, dt)

    def get_total_concentration(self) -> float:
        """Get total dye concentration across entire field"""
        return float(np.sum(self.concentration))

    def get_max_concentration(self) -> float:
        """Get maximum dye concentration in field"""
        return float(np.max(self.concentration))

    def get_concentration_stats(self) -> Dict[str, float]:
        """Get statistical summary of concentration field"""
        flat = self.concentration.flatten()
        return {
            'total': float(np.sum(flat)),
            'max': float(np.max(flat)),
            'mean': float(np.mean(flat)),
            'std': float(np.std(flat)),
            'nonzero_cells': int(np.count_nonzero(flat))
        }

    def clear_field(self):
        """Clear all dye from field"""
        self.concentration.fill(0.0)
        self.total_dye_injected = 0.0
        self.total_dye_decayed = 0.0

    def get_field_copy(self) -> np.ndarray:
        """Get copy of concentration field for analysis"""
        return self.concentration.copy()


class DyeSystem:
    """
    Complete dye system managing all three dye colors.

    Coordinates diffusion, decay, and injection across red, blue, and yellow
    dye fields with proper spatial and temporal dynamics.
    """

    def __init__(self, spatial_bounds: SpatialBounds):
        """
        Initialize dye system with spatial bounds.

        Args:
            spatial_bounds: Spatial bounds for all dye fields
        """
        self.spatial_bounds = spatial_bounds

        # Create dye fields for each color
        self.dye_fields = {
            DyeColor.RED: DyeField(DyeColor.RED, spatial_bounds),
            DyeColor.BLUE: DyeField(DyeColor.BLUE, spatial_bounds),
            DyeColor.YELLOW: DyeField(DyeColor.YELLOW, spatial_bounds)
        }

        # Track global statistics
        self.total_timesteps = 0

    def get_dye_field(self, dye_color: DyeColor) -> DyeField:
        """Get dye field for specific color"""
        return self.dye_fields[dye_color]

    def get_concentration(self, dye_color: DyeColor, activation: float, lag: float) -> float:
        """
        Get dye concentration for specific color at spatial location.

        Args:
            dye_color: Color of dye to query
            activation: Activation axis coordinate
            lag: Lag axis coordinate

        Returns:
            Dye concentration at specified location
        """
        return self.dye_fields[dye_color].get_concentration_at(activation, lag)

    def get_all_concentrations(self, activation: float, lag: float) -> Dict[DyeColor, float]:
        """Get concentrations of all dye colors at spatial location"""
        return {
            color: field.get_concentration_at(activation, lag)
            for color, field in self.dye_fields.items()
        }

    def inject_dye(self, dye_color: DyeColor, activation: float, lag: float,
                   amount: float, spread_radius: float = 1.0):
        """
        Inject dye of specified color at spatial location.

        Args:
            dye_color: Color of dye to inject
            activation: Activation axis coordinate
            lag: Lag axis coordinate
            amount: Amount of dye to inject
            spread_radius: Spatial spread radius
        """
        self.dye_fields[dye_color].inject_dye(activation, lag, amount, spread_radius)

    def inject_learning_signal(self, block_color: BlockColor, activation: float, lag: float,
                             config: DyeConfig, success: bool = True):
        """
        Inject learning signal dye based on trial outcome.

        Args:
            block_color: Color of block/wire creating learning signal
            activation: Spatial activation coordinate
            lag: Spatial lag coordinate
            config: Dye configuration
            success: Whether trial was successful (affects amount)
        """
        dye_color = DyeColor.from_block_color(block_color)

        # Determine injection amount based on success/failure
        if success:
            amount = config.injection_amount
        else:
            amount = config.injection_amount * 0.3  # Reduced for failed trials

        # Inject with spatial spread
        spread_radius = 2.0 * self.spatial_bounds.grid_resolution
        self.inject_dye(dye_color, activation, lag, amount, spread_radius)

    def step_all_dynamics(self, config: DyeConfig, dt: float):
        """
        Step dynamics for all dye fields.

        Args:
            config: Dye configuration parameters
            dt: Time step size
        """
        for field in self.dye_fields.values():
            field.step_dynamics(config, dt)

        self.total_timesteps += 1

    def get_system_stats(self) -> Dict[str, any]:
        """Get comprehensive system statistics"""
        stats = {
            'timesteps': self.total_timesteps,
            'spatial_bounds': {
                'activation_range': (self.spatial_bounds.activation_min,
                                   self.spatial_bounds.activation_max),
                'lag_range': (self.spatial_bounds.lag_min,
                            self.spatial_bounds.lag_max),
                'resolution': self.spatial_bounds.grid_resolution,
                'grid_shape': self.spatial_bounds.get_grid_shape()
            },
            'dye_fields': {}
        }

        for color, field in self.dye_fields.items():
            field_stats = field.get_concentration_stats()
            field_stats['total_injected'] = field.total_dye_injected
            field_stats['total_decayed'] = field.total_dye_decayed
            stats['dye_fields'][color.value] = field_stats

        return stats

    def clear_all_dye(self):
        """Clear all dye from all fields"""
        for field in self.dye_fields.values():
            field.clear_field()

    def get_field_snapshot(self) -> Dict[DyeColor, np.ndarray]:
        """Get snapshot of all concentration fields"""
        return {
            color: field.get_field_copy()
            for color, field in self.dye_fields.items()
        }

    def get_wire_enhancement_factor(self, wire_color: BlockColor, activation: float,
                                  lag: float, base_enhancement: float = 1.0) -> float:
        """
        Get dye enhancement factor for wire at spatial location.

        Args:
            wire_color: Color of wire to enhance
            activation: Wire spatial activation coordinate
            lag: Wire spatial lag coordinate
            base_enhancement: Base enhancement factor from config

        Returns:
            Enhancement factor for wire strength calculation
        """
        # Convert wire color to dye color
        try:
            dye_color = DyeColor.from_block_color(wire_color)
        except ValueError:
            return 0.0  # No enhancement for unknown colors

        # Get local dye concentration
        concentration = self.get_concentration(dye_color, activation, lag)

        # Calculate enhancement: 1 + α × C_local
        enhancement = 1.0 + base_enhancement * concentration

        return enhancement


def create_dye_system_for_network(activation_range: Tuple[float, float],
                                lag_range: Tuple[float, float],
                                resolution: float = 1.0) -> DyeSystem:
    """
    Factory function to create dye system for neural network.

    Args:
        activation_range: (min, max) activation axis bounds
        lag_range: (min, max) lag axis bounds
        resolution: Spatial grid resolution

    Returns:
        Initialized DyeSystem
    """
    spatial_bounds = SpatialBounds(
        activation_min=activation_range[0],
        activation_max=activation_range[1],
        lag_min=lag_range[0],
        lag_max=lag_range[1],
        grid_resolution=resolution
    )

    return DyeSystem(spatial_bounds)


if __name__ == "__main__":
    # Test dye system functionality
    from neural_klotski.config import get_default_config

    print("Testing Neural-Klotski Dye System...")

    # Create test dye system
    spatial_bounds = SpatialBounds(-50, 50, 0, 200, 2.0)
    dye_system = DyeSystem(spatial_bounds)

    config = get_default_config().dyes

    print(f"Grid shape: {spatial_bounds.get_grid_shape()}")
    print(f"Initial stats: {dye_system.get_system_stats()}")

    # Test dye injection
    dye_system.inject_dye(DyeColor.RED, 10.0, 50.0, 5.0)
    dye_system.inject_dye(DyeColor.BLUE, -10.0, 100.0, 3.0)

    print(f"After injection: {dye_system.get_system_stats()}")

    # Test concentration lookup
    red_conc = dye_system.get_concentration(DyeColor.RED, 10.0, 50.0)
    blue_conc = dye_system.get_concentration(DyeColor.BLUE, -10.0, 100.0)
    print(f"Red concentration at (10, 50): {red_conc:.3f}")
    print(f"Blue concentration at (-10, 100): {blue_conc:.3f}")

    # Test dynamics
    print(f"\nRunning 10 timesteps of diffusion and decay...")
    for step in range(10):
        dye_system.step_all_dynamics(config, 0.5)
        if step % 3 == 0:
            stats = dye_system.get_system_stats()
            total_red = stats['dye_fields']['red']['total']
            total_blue = stats['dye_fields']['blue']['total']
            print(f"Step {step}: Red={total_red:.3f}, Blue={total_blue:.3f}")

    # Test learning signal injection
    dye_system.inject_learning_signal(
        BlockColor.YELLOW, 0.0, 150.0, config, success=True
    )

    final_stats = dye_system.get_system_stats()
    print(f"\nFinal stats: {final_stats}")

    print("Dye system test completed successfully!")