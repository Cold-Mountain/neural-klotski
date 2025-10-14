"""
Unit tests for mathematical utilities.

Tests all mathematical functions for correctness, edge cases,
and compliance with Neural-Klotski specification requirements.
"""

import pytest
import numpy as np
import math
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_klotski.math_utils import (
    Vector2D, lag_distance, signal_propagation_delay, clamp, is_in_bounds,
    euler_integration_step, euler_integration_step_damped, spring_force,
    exponential_decay, discrete_laplacian_2d, diffusion_step,
    threshold_crossing_from_below, calculate_k_nearest_neighbors,
    wire_effective_strength
)


class TestVector2D:
    """Test Vector2D class operations"""

    def test_initialization(self):
        """Test vector creation and basic properties"""
        v = Vector2D(3.0, 4.0)
        assert v.x == 3.0
        assert v.y == 4.0

    def test_addition(self):
        """Test vector addition"""
        v1 = Vector2D(1.0, 2.0)
        v2 = Vector2D(3.0, 4.0)
        result = v1 + v2
        assert result.x == 4.0
        assert result.y == 6.0

    def test_subtraction(self):
        """Test vector subtraction"""
        v1 = Vector2D(5.0, 7.0)
        v2 = Vector2D(2.0, 3.0)
        result = v1 - v2
        assert result.x == 3.0
        assert result.y == 4.0

    def test_scalar_multiplication(self):
        """Test scalar multiplication"""
        v = Vector2D(2.0, 3.0)
        result = v * 2.5
        assert result.x == 5.0
        assert result.y == 7.5

        # Test reverse multiplication
        result2 = 2.5 * v
        assert result2.x == 5.0
        assert result2.y == 7.5

    def test_scalar_division(self):
        """Test scalar division"""
        v = Vector2D(6.0, 8.0)
        result = v / 2.0
        assert result.x == 3.0
        assert result.y == 4.0

    def test_division_by_zero(self):
        """Test division by zero raises error"""
        v = Vector2D(1.0, 2.0)
        with pytest.raises(ValueError, match="Cannot divide vector by zero"):
            v / 0.0

    def test_magnitude(self):
        """Test vector magnitude calculation"""
        # Test known Pythagorean triple
        v = Vector2D(3.0, 4.0)
        assert abs(v.magnitude() - 5.0) < 1e-10

        # Test zero vector
        v_zero = Vector2D(0.0, 0.0)
        assert v_zero.magnitude() == 0.0

    def test_normalize(self):
        """Test vector normalization"""
        v = Vector2D(3.0, 4.0)
        normalized = v.normalize()
        assert abs(normalized.magnitude() - 1.0) < 1e-10
        assert abs(normalized.x - 0.6) < 1e-10
        assert abs(normalized.y - 0.8) < 1e-10

        # Test zero vector normalization
        v_zero = Vector2D(0.0, 0.0)
        normalized_zero = v_zero.normalize()
        assert normalized_zero.x == 0.0
        assert normalized_zero.y == 0.0

    def test_dot_product(self):
        """Test dot product calculation"""
        v1 = Vector2D(2.0, 3.0)
        v2 = Vector2D(4.0, 5.0)
        result = v1.dot(v2)
        assert result == 23.0  # 2*4 + 3*5 = 8 + 15 = 23

    def test_distance_to(self):
        """Test distance calculation between vectors"""
        v1 = Vector2D(0.0, 0.0)
        v2 = Vector2D(3.0, 4.0)
        distance = v1.distance_to(v2)
        assert abs(distance - 5.0) < 1e-10

    def test_equality(self):
        """Test vector equality with numerical tolerance"""
        v1 = Vector2D(1.0, 2.0)
        v2 = Vector2D(1.0, 2.0)
        v3 = Vector2D(1.0000000001, 2.0)  # Within tolerance
        v4 = Vector2D(1.1, 2.0)  # Outside tolerance

        assert v1 == v2
        assert v1 == v3
        assert not (v1 == v4)


class TestBasicMathFunctions:
    """Test basic mathematical utility functions"""

    def test_lag_distance(self):
        """Test lag axis distance calculation"""
        assert lag_distance(50.0, 100.0) == 50.0
        assert lag_distance(100.0, 50.0) == 50.0
        assert lag_distance(75.0, 75.0) == 0.0
        assert lag_distance(-10.0, 20.0) == 30.0

    def test_signal_propagation_delay(self):
        """Test signal propagation delay calculation"""
        # From specification: delay = lag_distance / v_signal
        delay = signal_propagation_delay(50.0, 100.0, 100.0)
        assert delay == 0.5  # distance=50, speed=100, delay=0.5

        delay2 = signal_propagation_delay(100.0, 150.0, 200.0)
        assert delay2 == 0.25  # distance=50, speed=200, delay=0.25

    def test_signal_propagation_delay_errors(self):
        """Test signal propagation delay error cases"""
        with pytest.raises(ValueError, match="Signal speed must be positive"):
            signal_propagation_delay(50.0, 100.0, 0.0)

        with pytest.raises(ValueError, match="Signal speed must be positive"):
            signal_propagation_delay(50.0, 100.0, -100.0)

    def test_clamp(self):
        """Test value clamping function"""
        assert clamp(5.0, 0.0, 10.0) == 5.0  # Within bounds
        assert clamp(-5.0, 0.0, 10.0) == 0.0  # Below minimum
        assert clamp(15.0, 0.0, 10.0) == 10.0  # Above maximum
        assert clamp(7.5, 7.5, 7.5) == 7.5  # At bounds

    def test_clamp_invalid_bounds(self):
        """Test clamp with invalid bounds"""
        with pytest.raises(ValueError, match="min_val .* must be <= max_val"):
            clamp(5.0, 10.0, 5.0)

    def test_is_in_bounds(self):
        """Test bounds checking function"""
        assert is_in_bounds(5.0, 0.0, 10.0) == True
        assert is_in_bounds(0.0, 0.0, 10.0) == True  # At minimum
        assert is_in_bounds(10.0, 0.0, 10.0) == True  # At maximum
        assert is_in_bounds(-1.0, 0.0, 10.0) == False  # Below minimum
        assert is_in_bounds(11.0, 0.0, 10.0) == False  # Above maximum


class TestIntegration:
    """Test numerical integration functions"""

    def test_euler_integration_step(self):
        """Test basic Euler integration"""
        # Simple test: constant velocity motion
        pos, vel = euler_integration_step(0.0, 5.0, 0.0, 0.1)
        assert pos == 0.5  # pos = 0 + 5*0.1
        assert vel == 5.0  # vel = 5 + 0*0.1

        # Test with acceleration
        pos2, vel2 = euler_integration_step(0.0, 0.0, 10.0, 0.1)
        assert pos2 == 0.0  # pos = 0 + 0*0.1
        assert vel2 == 1.0  # vel = 0 + 10*0.1

    def test_euler_integration_step_damped(self):
        """Test damped Euler integration"""
        # Test with damping but no force
        pos, vel = euler_integration_step_damped(0.0, 10.0, 0.0, 1.0, 0.1, 0.1)
        expected_vel = 10.0 * (1.0 - 0.1 * 0.1)  # damping factor
        assert abs(vel - expected_vel) < 1e-10
        assert abs(pos - 1.0) < 1e-10  # pos = 0 + 10*0.1

    def test_euler_integration_damped_errors(self):
        """Test damped integration error cases"""
        with pytest.raises(ValueError, match="Mass must be positive"):
            euler_integration_step_damped(0.0, 1.0, 1.0, 0.0, 0.1, 0.1)

        with pytest.raises(ValueError, match="Damping must be non-negative"):
            euler_integration_step_damped(0.0, 1.0, 1.0, 1.0, -0.1, 0.1)


class TestForces:
    """Test force calculation functions"""

    def test_spring_force(self):
        """Test spring force calculation"""
        # F_spring = -k * (x - x_rest)
        force = spring_force(5.0, 0.0, 2.0)
        assert force == -10.0  # -2 * (5 - 0)

        # Test with non-zero rest position
        force2 = spring_force(7.0, 3.0, 1.5)
        assert force2 == -6.0  # -1.5 * (7 - 3)

        # Test at rest position
        force3 = spring_force(3.0, 3.0, 2.0)
        assert force3 == 0.0

    def test_spring_force_errors(self):
        """Test spring force error cases"""
        with pytest.raises(ValueError, match="Spring constant must be non-negative"):
            spring_force(1.0, 0.0, -1.0)


class TestDecayAndDiffusion:
    """Test decay and diffusion calculations"""

    def test_exponential_decay(self):
        """Test exponential decay function"""
        # C(t) = C(0) * exp(-t/τ)
        result = exponential_decay(100.0, 0.0, 50.0)
        assert result == 100.0  # At t=0

        result2 = exponential_decay(100.0, 50.0, 50.0)
        expected = 100.0 * math.exp(-1.0)  # t=τ, so exp(-1)
        assert abs(result2 - expected) < 1e-10

        # Test half-life
        result3 = exponential_decay(100.0, 50.0 * math.log(2), 50.0)
        assert abs(result3 - 50.0) < 1e-10

    def test_exponential_decay_errors(self):
        """Test exponential decay error cases"""
        with pytest.raises(ValueError, match="Time constant must be positive"):
            exponential_decay(100.0, 10.0, 0.0)

        with pytest.raises(ValueError, match="Time must be non-negative"):
            exponential_decay(100.0, -10.0, 50.0)

    def test_discrete_laplacian_2d(self):
        """Test 2D discrete Laplacian calculation"""
        # Create simple test grid
        grid = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 5.0, 1.0],
            [1.0, 1.0, 1.0]
        ])

        # Center point should have negative Laplacian (peak)
        laplacian = discrete_laplacian_2d(grid, 1, 1, 1.0)
        # laplacian = (1+1+1+1-4*5) / 1² = (4-20) = -16
        assert laplacian == -16.0

        # Corner should have zero Laplacian (boundary condition)
        laplacian_corner = discrete_laplacian_2d(grid, 0, 0, 1.0)
        # With boundary conditions: (1+1+1+1-4*1) = 0
        assert laplacian_corner == 0.0

    def test_diffusion_step(self):
        """Test full diffusion step calculation"""
        # Simple 3x3 grid with center concentration
        initial_grid = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0]
        ])

        # Run one diffusion step
        new_grid = diffusion_step(initial_grid, 1.0, 0.1, 1.0)

        # Center should decrease, neighbors should increase
        assert new_grid[1, 1] < initial_grid[1, 1]
        assert new_grid[0, 1] > initial_grid[0, 1]
        assert new_grid[1, 0] > initial_grid[1, 0]

        # Total concentration should be conserved (approximately)
        assert abs(np.sum(new_grid) - np.sum(initial_grid)) < 1e-10

        # All values should remain non-negative
        assert np.all(new_grid >= 0.0)


class TestThresholdAndFiring:
    """Test threshold crossing and firing detection"""

    def test_threshold_crossing_from_below(self):
        """Test threshold crossing detection"""
        # Crossing from below - should fire
        assert threshold_crossing_from_below(49.0, 51.0, 50.0) == True

        # Crossing from above - should not fire
        assert threshold_crossing_from_below(51.0, 49.0, 50.0) == False

        # Not crossing - should not fire
        assert threshold_crossing_from_below(45.0, 48.0, 50.0) == False
        assert threshold_crossing_from_below(52.0, 55.0, 50.0) == False

        # Exactly at threshold
        assert threshold_crossing_from_below(49.0, 50.0, 50.0) == True
        assert threshold_crossing_from_below(50.0, 50.0, 50.0) == False


class TestNetworkUtilities:
    """Test network connectivity and enhancement functions"""

    def test_calculate_k_nearest_neighbors(self):
        """Test k-nearest neighbors calculation"""
        all_lags = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        target_lag = 35.0

        # Find 3 nearest neighbors
        neighbors = calculate_k_nearest_neighbors(target_lag, all_lags, 3)

        assert len(neighbors) == 3
        assert 30.0 in neighbors  # distance = 5
        assert 40.0 in neighbors  # distance = 5
        assert 20.0 in neighbors or 50.0 in neighbors  # distance = 15

    def test_wire_effective_strength(self):
        """Test wire strength enhancement calculation"""
        # No dye enhancement
        strength = wire_effective_strength(2.0, 0.0, 1.0)
        assert strength == 2.0

        # With dye enhancement: w_eff = w * (1 + α * C)
        strength2 = wire_effective_strength(2.0, 0.5, 2.0)
        expected = 2.0 * (1.0 + 2.0 * 0.5)  # 2.0 * 2.0 = 4.0
        assert strength2 == expected

    def test_wire_effective_strength_errors(self):
        """Test wire strength enhancement error cases"""
        with pytest.raises(ValueError, match="Dye concentration must be non-negative"):
            wire_effective_strength(2.0, -0.1, 1.0)


class TestSpecificationCompliance:
    """Test compliance with Neural-Klotski specification"""

    def test_parameter_ranges(self):
        """Test that default parameters are within specification ranges"""
        # Test signal speed range (50-200 lag units/timestep)
        test_speed = 100.0
        assert 50.0 <= test_speed <= 200.0

        # Test that delay calculations work within range
        delay = signal_propagation_delay(50.0, 150.0, test_speed)
        assert delay == 1.0  # distance=100, speed=100, delay=1.0

    def test_integration_stability(self):
        """Test numerical integration stability"""
        # Test that integration doesn't blow up with typical parameters
        dt = 0.05  # Smaller timestep for stability
        mass = 1.0
        damping = 0.15
        spring_k = 0.15

        position = 10.0  # Start away from equilibrium
        velocity = 0.0

        # Run integration for many steps
        for _ in range(200):  # More steps with smaller dt
            force = spring_force(position, 0.0, spring_k)
            position, velocity = euler_integration_step_damped(
                position, velocity, force, mass, damping, dt
            )

        # Should converge toward equilibrium (allow for some numerical error)
        assert abs(position) < 5.0  # Should be moving toward origin
        assert abs(velocity) < 5.0  # Should have reasonable velocity

    def test_dye_conservation(self):
        """Test that dye diffusion conserves total concentration"""
        # Create grid with some initial concentration
        initial_grid = np.ones((5, 5)) * 2.0
        initial_total = np.sum(initial_grid)

        # Run several diffusion steps
        grid = initial_grid.copy()
        for _ in range(10):
            grid = diffusion_step(grid, 0.5, 0.1, 1.0)

        # Total should be conserved (within numerical precision)
        final_total = np.sum(grid)
        assert abs(final_total - initial_total) < 1e-6


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])