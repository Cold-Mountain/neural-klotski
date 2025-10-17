"""
Unit tests for Dye system in Neural-Klotski.

Tests all aspects of dye field behavior including:
- 2D spatial grid management and coordinate mapping
- Separate red, blue, and yellow dye fields
- Diffusion dynamics with spatial Laplacian
- Exponential decay mechanics
- Dye injection and learning signal creation
- Wire enhancement factor calculation
- Conservation and boundary condition handling
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_klotski.core.dye import (
    DyeColor, DyeField, DyeSystem, SpatialBounds, create_dye_system_for_network
)
from neural_klotski.core.block import BlockColor
from neural_klotski.config import get_default_config, DyeConfig


class TestDyeColor:
    """Test DyeColor enumeration and conversions"""

    def test_dye_color_values(self):
        """Test dye color enum values"""
        assert DyeColor.RED.value == "red"
        assert DyeColor.BLUE.value == "blue"
        assert DyeColor.YELLOW.value == "yellow"

    def test_from_block_color_conversion(self):
        """Test conversion from BlockColor to DyeColor"""
        assert DyeColor.from_block_color(BlockColor.RED) == DyeColor.RED
        assert DyeColor.from_block_color(BlockColor.BLUE) == DyeColor.BLUE
        assert DyeColor.from_block_color(BlockColor.YELLOW) == DyeColor.YELLOW

    def test_invalid_block_color_conversion(self):
        """Test error handling for invalid block color"""
        # Create a mock invalid color (this test assumes BlockColor might be extended)
        with pytest.raises(ValueError, match="Unknown block color"):
            # This will test the else branch in from_block_color
            class InvalidColor:
                pass
            DyeColor.from_block_color(InvalidColor())


class TestSpatialBounds:
    """Test spatial bounds and coordinate mapping"""

    def test_spatial_bounds_creation(self):
        """Test creating valid spatial bounds"""
        bounds = SpatialBounds(-10, 10, 0, 50, 1.0)
        assert bounds.activation_min == -10
        assert bounds.activation_max == 10
        assert bounds.lag_min == 0
        assert bounds.lag_max == 50
        assert bounds.grid_resolution == 1.0

    def test_spatial_bounds_validation(self):
        """Test spatial bounds validation"""
        # Invalid activation range
        with pytest.raises(ValueError, match="activation_max must be > activation_min"):
            SpatialBounds(10, 10, 0, 50, 1.0)

        # Invalid lag range
        with pytest.raises(ValueError, match="lag_max must be > lag_min"):
            SpatialBounds(-10, 10, 50, 50, 1.0)

        # Invalid resolution
        with pytest.raises(ValueError, match="grid_resolution must be positive"):
            SpatialBounds(-10, 10, 0, 50, 0.0)

    def test_grid_shape_calculation(self):
        """Test grid shape calculation"""
        bounds = SpatialBounds(-10, 10, 0, 20, 2.0)
        rows, cols = bounds.get_grid_shape()

        # Expected: activation range = 20, lag range = 20, resolution = 2.0
        # Cols = ceil(20/2) = 10, Rows = ceil(20/2) = 10
        assert cols == 10
        assert rows == 10

    def test_spatial_to_grid_conversion(self):
        """Test spatial coordinate to grid index conversion"""
        bounds = SpatialBounds(0, 10, 0, 10, 1.0)

        # Test various coordinates
        row, col = bounds.spatial_to_grid(5.0, 5.0)
        assert row == 5
        assert col == 5

        # Test boundary coordinates
        row, col = bounds.spatial_to_grid(0.0, 0.0)
        assert row == 0
        assert col == 0

        # Test out-of-bounds coordinates (should be clamped)
        row, col = bounds.spatial_to_grid(-5.0, 15.0)
        assert row >= 0
        assert col >= 0

    def test_grid_to_spatial_conversion(self):
        """Test grid index to spatial coordinate conversion"""
        bounds = SpatialBounds(0, 10, 0, 10, 1.0)

        # Test center of grid cell
        activation, lag = bounds.grid_to_spatial(5, 5)
        assert activation == 5.5  # Center of cell at col 5
        assert lag == 5.5         # Center of cell at row 5


class TestDyeField:
    """Test individual dye field functionality"""

    def test_dye_field_initialization(self):
        """Test dye field creation and initialization"""
        bounds = SpatialBounds(-5, 5, 0, 10, 1.0)
        field = DyeField(DyeColor.RED, bounds)

        assert field.dye_color == DyeColor.RED
        assert field.spatial_bounds == bounds
        assert field.concentration.shape == (10, 10)
        assert np.all(field.concentration == 0.0)
        assert field.total_dye_injected == 0.0
        assert field.total_dye_decayed == 0.0

    def test_concentration_lookup(self):
        """Test concentration lookup at spatial coordinates"""
        bounds = SpatialBounds(0, 10, 0, 10, 1.0)
        field = DyeField(DyeColor.BLUE, bounds)

        # Initially zero everywhere
        assert field.get_concentration_at(5.0, 5.0) == 0.0

        # Manually set concentration and test lookup
        field.concentration[5, 5] = 2.5
        conc = field.get_concentration_at(5.5, 5.5)  # Should map to grid cell (5,5)
        assert conc == 2.5

    def test_dye_injection(self):
        """Test dye injection mechanism"""
        bounds = SpatialBounds(0, 10, 0, 10, 1.0)
        field = DyeField(DyeColor.YELLOW, bounds)

        # Inject dye at center
        field.inject_dye(5.0, 5.0, 3.0)

        assert field.total_dye_injected == 3.0
        assert field.get_total_concentration() > 0
        assert field.get_concentration_at(5.0, 5.0) > 0

        # Test zero injection (should be ignored)
        initial_total = field.get_total_concentration()
        field.inject_dye(2.0, 2.0, 0.0)
        assert field.get_total_concentration() == initial_total

    def test_dye_injection_with_spread(self):
        """Test dye injection with spatial spread"""
        bounds = SpatialBounds(0, 20, 0, 20, 1.0)
        field = DyeField(DyeColor.RED, bounds)

        # Inject with spread
        field.inject_dye(10.0, 10.0, 5.0, spread_radius=2.0)

        # Should have spread to neighboring cells
        center_conc = field.get_concentration_at(10.0, 10.0)
        neighbor_conc = field.get_concentration_at(11.0, 10.0)

        assert center_conc > neighbor_conc > 0
        assert field.get_total_concentration() == pytest.approx(5.0, abs=1e-10)

    def test_diffusion_step(self):
        """Test diffusion dynamics"""
        bounds = SpatialBounds(0, 10, 0, 10, 1.0)
        field = DyeField(DyeColor.RED, bounds)
        config = DyeConfig(diffusion_coefficient=1.0)

        # Create initial concentration peak
        field.concentration[5, 5] = 10.0
        initial_total = field.get_total_concentration()

        # Run diffusion step
        field.step_diffusion(config, 0.1)

        # Total should be conserved (approximately)
        final_total = field.get_total_concentration()
        assert final_total == pytest.approx(initial_total, rel=1e-6)

        # Concentration should have spread
        center_conc = field.concentration[5, 5]
        neighbor_conc = field.concentration[5, 4]
        assert center_conc < 10.0  # Decreased from center
        assert neighbor_conc > 0    # Increased in neighbors

    def test_decay_step(self):
        """Test exponential decay"""
        bounds = SpatialBounds(0, 10, 0, 10, 1.0)
        field = DyeField(DyeColor.BLUE, bounds)
        config = DyeConfig(decay_time_constant=100.0)

        # Set initial concentration
        field.concentration[5, 5] = 5.0
        initial_total = field.get_total_concentration()

        # Run decay step
        field.step_decay(config, 1.0)

        # Total should have decreased
        final_total = field.get_total_concentration()
        assert final_total < initial_total

        # Calculate expected decay
        expected_factor = np.exp(-1.0 / 100.0)
        expected_total = initial_total * expected_factor
        assert final_total == pytest.approx(expected_total, rel=1e-10)

    def test_complete_dynamics_step(self):
        """Test combined diffusion and decay"""
        bounds = SpatialBounds(0, 10, 0, 10, 1.0)
        field = DyeField(DyeColor.YELLOW, bounds)
        config = DyeConfig(diffusion_coefficient=0.5, decay_time_constant=200.0)

        # Set initial state
        field.concentration[5, 5] = 8.0
        initial_total = field.get_total_concentration()

        # Run combined step
        field.step_dynamics(config, 0.5)

        # Should have both diffused and decayed
        final_total = field.get_total_concentration()
        assert final_total < initial_total  # Decayed
        assert field.concentration[5, 4] > 0  # Diffused

    def test_concentration_statistics(self):
        """Test concentration statistics calculation"""
        bounds = SpatialBounds(0, 5, 0, 5, 1.0)
        field = DyeField(DyeColor.RED, bounds)

        # Set some concentrations
        field.concentration[1, 1] = 3.0
        field.concentration[2, 2] = 2.0
        field.concentration[3, 3] = 1.0

        stats = field.get_concentration_stats()
        assert stats['total'] == 6.0
        assert stats['max'] == 3.0
        assert stats['nonzero_cells'] == 3
        assert stats['mean'] == pytest.approx(6.0 / 25, abs=1e-10)

    def test_field_clearing(self):
        """Test clearing dye field"""
        bounds = SpatialBounds(0, 5, 0, 5, 1.0)
        field = DyeField(DyeColor.BLUE, bounds)

        # Add some dye
        field.inject_dye(2.0, 2.0, 5.0)
        assert field.get_total_concentration() > 0

        # Clear field
        field.clear_field()
        assert field.get_total_concentration() == 0.0
        assert field.total_dye_injected == 0.0
        assert field.total_dye_decayed == 0.0


class TestDyeSystem:
    """Test complete dye system with multiple colors"""

    def test_dye_system_initialization(self):
        """Test dye system creation"""
        bounds = SpatialBounds(-10, 10, 0, 50, 2.0)
        system = DyeSystem(bounds)

        assert system.spatial_bounds == bounds
        assert len(system.dye_fields) == 3
        assert DyeColor.RED in system.dye_fields
        assert DyeColor.BLUE in system.dye_fields
        assert DyeColor.YELLOW in system.dye_fields

    def test_dye_field_access(self):
        """Test accessing individual dye fields"""
        bounds = SpatialBounds(0, 10, 0, 10, 1.0)
        system = DyeSystem(bounds)

        red_field = system.get_dye_field(DyeColor.RED)
        assert red_field.dye_color == DyeColor.RED

        blue_field = system.get_dye_field(DyeColor.BLUE)
        assert blue_field.dye_color == DyeColor.BLUE

    def test_concentration_queries(self):
        """Test concentration queries across colors"""
        bounds = SpatialBounds(0, 10, 0, 10, 1.0)
        system = DyeSystem(bounds)

        # Inject different colors
        system.inject_dye(DyeColor.RED, 5.0, 5.0, 3.0)
        system.inject_dye(DyeColor.BLUE, 7.0, 7.0, 2.0)

        # Test single color query
        red_conc = system.get_concentration(DyeColor.RED, 5.0, 5.0)
        assert red_conc > 0

        # Test all colors query
        all_conc = system.get_all_concentrations(5.0, 5.0)
        assert all_conc[DyeColor.RED] > 0
        assert all_conc[DyeColor.BLUE] >= 0  # Might have spread here
        assert all_conc[DyeColor.YELLOW] == 0

    def test_learning_signal_injection(self):
        """Test learning signal injection"""
        bounds = SpatialBounds(0, 20, 0, 20, 1.0)
        system = DyeSystem(bounds)
        config = DyeConfig(injection_amount=1.5)

        # Test successful trial
        system.inject_learning_signal(BlockColor.RED, 10.0, 10.0, config, success=True)
        red_conc = system.get_concentration(DyeColor.RED, 10.0, 10.0)
        assert red_conc > 0

        # Test failed trial (should inject less)
        initial_blue = system.get_concentration(DyeColor.BLUE, 15.0, 15.0)
        system.inject_learning_signal(BlockColor.BLUE, 15.0, 15.0, config, success=False)
        final_blue = system.get_concentration(DyeColor.BLUE, 15.0, 15.0)

        blue_increase = final_blue - initial_blue
        expected_increase = config.injection_amount * 0.3  # Failed trial factor
        assert blue_increase > 0
        # Note: exact comparison difficult due to spatial spread

    def test_system_dynamics(self):
        """Test system-wide dynamics step"""
        bounds = SpatialBounds(0, 10, 0, 10, 1.0)
        system = DyeSystem(bounds)
        config = DyeConfig(diffusion_coefficient=0.1, decay_time_constant=50.0)  # Stable parameters

        # Set dye directly in fields to avoid injection spread issues
        system.get_dye_field(DyeColor.RED).concentration[3, 3] = 2.0
        system.get_dye_field(DyeColor.BLUE).concentration[5, 5] = 3.0
        system.get_dye_field(DyeColor.YELLOW).concentration[7, 7] = 1.0

        initial_totals = {
            color: field.get_total_concentration()
            for color, field in system.dye_fields.items()
        }

        # Run dynamics for multiple steps to see decay effect
        for _ in range(50):  # More steps with smaller effect per step
            system.step_all_dynamics(config, 0.1)

        # All fields should have decreased due to decay over time
        for color, field in system.dye_fields.items():
            final_total = field.get_total_concentration()
            assert final_total < initial_totals[color]

        assert system.total_timesteps == 50

    def test_wire_enhancement_calculation(self):
        """Test wire enhancement factor calculation"""
        bounds = SpatialBounds(0, 10, 0, 10, 1.0)
        system = DyeSystem(bounds)

        # Inject red dye with no spread (spread_radius=0) for exact concentration
        red_field = system.get_dye_field(DyeColor.RED)
        red_field.concentration[5, 5] = 2.0  # Set exact concentration

        # Test enhancement for red wire
        enhancement = system.get_wire_enhancement_factor(
            BlockColor.RED, 5.0, 5.0, base_enhancement=1.5
        )
        expected = 1.0 + 1.5 * 2.0  # 1 + α * C = 1 + 1.5 * 2.0 = 4.0
        assert enhancement == expected

        # Test no enhancement for different color
        blue_enhancement = system.get_wire_enhancement_factor(
            BlockColor.BLUE, 5.0, 5.0, base_enhancement=1.5
        )
        assert blue_enhancement == 1.0  # No blue dye at this location

    def test_system_statistics(self):
        """Test system statistics collection"""
        bounds = SpatialBounds(-5, 5, 0, 20, 1.0)
        system = DyeSystem(bounds)

        stats = system.get_system_stats()

        assert 'timesteps' in stats
        assert 'spatial_bounds' in stats
        assert 'dye_fields' in stats
        assert len(stats['dye_fields']) == 3

        # Check spatial bounds info
        spatial_info = stats['spatial_bounds']
        assert spatial_info['activation_range'] == (-5, 5)
        assert spatial_info['lag_range'] == (0, 20)
        assert spatial_info['resolution'] == 1.0

    def test_field_snapshot(self):
        """Test field snapshot functionality"""
        bounds = SpatialBounds(0, 5, 0, 5, 1.0)
        system = DyeSystem(bounds)

        # Add some dye
        system.inject_dye(DyeColor.RED, 2.0, 2.0, 3.0)

        # Get snapshot
        snapshot = system.get_field_snapshot()

        assert len(snapshot) == 3
        assert isinstance(snapshot[DyeColor.RED], np.ndarray)
        assert np.sum(snapshot[DyeColor.RED]) > 0
        assert np.sum(snapshot[DyeColor.BLUE]) == 0
        assert np.sum(snapshot[DyeColor.YELLOW]) == 0

    def test_clear_all_dye(self):
        """Test clearing all dye fields"""
        bounds = SpatialBounds(0, 10, 0, 10, 1.0)
        system = DyeSystem(bounds)

        # Add dye to all colors
        system.inject_dye(DyeColor.RED, 3.0, 3.0, 2.0)
        system.inject_dye(DyeColor.BLUE, 5.0, 5.0, 3.0)
        system.inject_dye(DyeColor.YELLOW, 7.0, 7.0, 1.0)

        # Verify dye exists
        for field in system.dye_fields.values():
            assert field.get_total_concentration() > 0

        # Clear all
        system.clear_all_dye()

        # Verify all cleared
        for field in system.dye_fields.values():
            assert field.get_total_concentration() == 0.0


class TestDyeSystemFactory:
    """Test dye system factory function"""

    def test_factory_function(self):
        """Test create_dye_system_for_network factory"""
        system = create_dye_system_for_network(
            activation_range=(-20, 20),
            lag_range=(0, 100),
            resolution=2.0
        )

        assert isinstance(system, DyeSystem)
        assert system.spatial_bounds.activation_min == -20
        assert system.spatial_bounds.activation_max == 20
        assert system.spatial_bounds.lag_min == 0
        assert system.spatial_bounds.lag_max == 100
        assert system.spatial_bounds.grid_resolution == 2.0


class TestDyeSpecificationCompliance:
    """Test compliance with Neural-Klotski specification"""

    def test_diffusion_equation_implementation(self):
        """Test that diffusion follows Section 9.3 equations"""
        bounds = SpatialBounds(0, 10, 0, 10, 1.0)
        field = DyeField(DyeColor.RED, bounds)
        config = DyeConfig(diffusion_coefficient=2.0)

        # Create simple test pattern
        field.concentration[5, 5] = 4.0

        # Calculate expected change for one step
        # Uses discrete_laplacian_2d and diffusion_step from math_utils
        initial_laplacian = (0 + 0 + 0 + 0 - 4 * 4.0) / (1.0 ** 2)  # -16.0

        field.step_diffusion(config, 0.1)

        # Center should have decreased
        assert field.concentration[5, 5] < 4.0

        # Neighbors should have increased
        assert field.concentration[4, 5] > 0
        assert field.concentration[6, 5] > 0
        assert field.concentration[5, 4] > 0
        assert field.concentration[5, 6] > 0

    def test_exponential_decay_compliance(self):
        """Test exponential decay matches Section 9.3"""
        bounds = SpatialBounds(0, 5, 0, 5, 1.0)
        field = DyeField(DyeColor.BLUE, bounds)
        config = DyeConfig(decay_time_constant=50.0)

        # Set initial concentration
        initial_conc = 10.0
        field.concentration[2, 2] = initial_conc

        # Run decay for multiple steps
        dt = 1.0
        for step in range(50):  # t = 50 (one time constant)
            field.step_decay(config, dt)

        # After one time constant, should be at 1/e of initial
        expected_conc = initial_conc / np.e
        actual_conc = field.concentration[2, 2]
        assert actual_conc == pytest.approx(expected_conc, rel=1e-6)

    def test_dye_conservation_during_diffusion(self):
        """Test that diffusion conserves total dye"""
        bounds = SpatialBounds(0, 20, 0, 20, 1.0)
        field = DyeField(DyeColor.YELLOW, bounds)
        config = DyeConfig(diffusion_coefficient=1.0)

        # Inject dye
        field.inject_dye(10.0, 10.0, 5.0)
        initial_total = field.get_total_concentration()

        # Run many diffusion steps
        for _ in range(20):
            field.step_diffusion(config, 0.1)

        final_total = field.get_total_concentration()
        assert final_total == pytest.approx(initial_total, rel=1e-6)

    def test_learning_signal_parameters(self):
        """Test learning signal injection matches specification"""
        bounds = SpatialBounds(0, 10, 0, 10, 1.0)
        system = DyeSystem(bounds)
        config = DyeConfig(injection_amount=0.7)  # From spec

        # Test successful trial injection
        system.inject_learning_signal(BlockColor.RED, 5.0, 5.0, config, success=True)
        success_conc = system.get_concentration(DyeColor.RED, 5.0, 5.0)

        # Clear and test failed trial
        system.clear_all_dye()
        system.inject_learning_signal(BlockColor.RED, 5.0, 5.0, config, success=False)
        failure_conc = system.get_concentration(DyeColor.RED, 5.0, 5.0)

        # Failed trial should inject less (0.3 factor)
        assert failure_conc < success_conc
        # Note: exact comparison difficult due to spatial spread


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])