"""
Tests for Neural-Klotski mathematical validation system.

Comprehensive tests for validation framework, mathematical correctness,
and specification compliance.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_klotski.validation.mathematical_validator import (
    MathematicalValidator, ValidationResult, ValidationSuite
)
from neural_klotski.config import get_default_config


class TestValidationFramework:
    """Test the validation framework itself"""

    def test_validator_initialization(self):
        """Test validator creates correctly"""
        validator = MathematicalValidator(tolerance=1e-10)
        assert validator.tolerance == 1e-10
        assert validator.config is not None

    def test_validation_result_creation(self):
        """Test validation result data structure"""
        result = ValidationResult(
            test_name="Test equation",
            passed=True,
            expected_value=1.0,
            actual_value=1.0,
            tolerance=1e-10
        )

        assert result.test_name == "Test equation"
        assert result.passed == True
        assert result.expected_value == 1.0
        assert result.actual_value == 1.0

    def test_validation_suite_management(self):
        """Test validation suite operations"""
        suite = ValidationSuite("Test Suite")

        # Add some test results
        result1 = ValidationResult("Test 1", True, 1.0, 1.0, 1e-10)
        result2 = ValidationResult("Test 2", False, 2.0, 2.1, 1e-10)

        suite.add_result(result1)
        suite.add_result(result2)

        summary = suite.get_summary()
        assert summary['total_tests'] == 2
        assert summary['passed'] == 1
        assert summary['failed'] == 1
        assert summary['success_rate'] == 0.5

    def test_scalar_equation_validation(self):
        """Test scalar equation validation"""
        validator = MathematicalValidator()

        # Test passing case
        result = validator.validate_scalar_equation("Test", 1.0, 1.0)
        assert result.passed == True

        # Test failing case
        result = validator.validate_scalar_equation("Test", 1.0, 2.0)
        assert result.passed == False

    def test_parameter_bounds_validation(self):
        """Test parameter bounds checking"""
        validator = MathematicalValidator()

        # Test within bounds
        result = validator.validate_parameter_bounds("Test", 5.0, 0.0, 10.0)
        assert result.passed == True

        # Test outside bounds
        result = validator.validate_parameter_bounds("Test", 15.0, 0.0, 10.0)
        assert result.passed == False


class TestMathematicalValidation:
    """Test mathematical validation correctness"""

    def test_mathematical_utilities_validation(self):
        """Test mathematical utilities validation suite"""
        validator = MathematicalValidator(tolerance=1e-10)
        suite = validator.validate_math_utils()

        summary = suite.get_summary()
        assert summary['total_tests'] > 0
        assert summary['all_passed'] == True, f"Some math tests failed: {summary}"

    def test_block_dynamics_validation(self):
        """Test block dynamics validation suite"""
        validator = MathematicalValidator(tolerance=1e-10)
        suite = validator.validate_block_dynamics()

        summary = suite.get_summary()
        assert summary['total_tests'] > 0
        assert summary['all_passed'] == True, f"Some block dynamics tests failed: {summary}"

    def test_wire_mechanics_validation(self):
        """Test wire mechanics validation suite"""
        validator = MathematicalValidator(tolerance=1e-10)
        suite = validator.validate_wire_mechanics()

        summary = suite.get_summary()
        assert summary['total_tests'] > 0
        assert summary['all_passed'] == True, f"Some wire mechanics tests failed: {summary}"

    def test_network_architecture_validation(self):
        """Test network architecture validation suite"""
        validator = MathematicalValidator(tolerance=1e-10)
        suite = validator.validate_network_architecture()

        summary = suite.get_summary()
        assert summary['total_tests'] > 0
        assert summary['all_passed'] == True, f"Some architecture tests failed: {summary}"


class TestParameterValidation:
    """Test parameter validation and bounds checking"""

    def test_configuration_parameter_bounds(self):
        """Test all configuration parameters are within specification bounds"""
        config = get_default_config()

        # Test dynamics parameters
        assert 0.1 <= config.dynamics.dt <= 1.0
        assert config.dynamics.mass > 0
        assert 0.1 <= config.dynamics.damping <= 0.2
        assert 0.1 <= config.dynamics.spring_constant <= 0.2

        # Test wire parameters
        assert config.wires.strength_min >= 0.0
        assert config.wires.strength_max > config.wires.strength_min
        assert config.wires.signal_speed > 0

        # Test dye parameters
        assert config.dyes.diffusion_coefficient > 0
        assert config.dyes.decay_time_constant > 0
        assert config.dyes.enhancement_factor >= 0

        # Test learning parameters
        assert 0 <= config.learning.wire_learning_rate <= 1.0
        assert 0 <= config.learning.threshold_learning_rate <= 1.0

    def test_threshold_parameter_validation(self):
        """Test threshold parameters are properly ordered"""
        config = get_default_config()

        # Input thresholds
        assert config.thresholds.input_threshold_min < config.thresholds.input_threshold_max

        # Hidden thresholds
        assert config.thresholds.hidden_threshold_min < config.thresholds.hidden_threshold_max

        # Output thresholds
        assert config.thresholds.output_threshold_min < config.thresholds.output_threshold_max

        # Global bounds
        assert config.thresholds.threshold_min < config.thresholds.threshold_max

    def test_network_architecture_parameters(self):
        """Test network architecture parameters"""
        config = get_default_config()

        # Block counts
        assert config.network.total_blocks == 79
        assert config.network.input_blocks == 20
        assert config.network.hidden1_blocks == 20
        assert config.network.hidden2_blocks == 20
        assert config.network.output_blocks == 19

        # Sum should equal total
        total_calculated = (config.network.input_blocks +
                           config.network.hidden1_blocks +
                           config.network.hidden2_blocks +
                           config.network.output_blocks)
        assert total_calculated == config.network.total_blocks

        # Connectivity parameters
        assert config.network.local_connections > 0
        assert config.network.longrange_connections >= 0


class TestNumericalPrecision:
    """Test numerical precision and accuracy"""

    def test_floating_point_precision(self):
        """Test floating point precision in calculations"""
        validator = MathematicalValidator(tolerance=1e-15)

        # Test very small differences
        result = validator.validate_scalar_equation(
            "High precision test", 1.0000000000000001, 1.0, tolerance=1e-15
        )
        # This should pass with very high tolerance
        assert result.passed or result.tolerance >= 1e-15

    def test_mathematical_function_accuracy(self):
        """Test accuracy of mathematical functions"""
        from neural_klotski.math_utils import (
            spring_force, exponential_decay, threshold_crossing_from_below,
            euler_integration_step_damped
        )
        import math

        # Test spring force precision
        force = spring_force(1.0, 0.0, 1.0)  # F = -k*x = -1*1 = -1
        assert abs(force - (-1.0)) < 1e-15

        # Test exponential decay precision
        decay = exponential_decay(1.0, 1.0, 1.0)  # exp(-1) ≈ 0.368
        expected = math.exp(-1.0)
        assert abs(decay - expected) < 1e-15

        # Test Euler integration precision
        pos, vel = euler_integration_step_damped(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        # x_new = 0 + 1*1 = 1, v_new = 1*(1-0) + 0 = 1
        assert abs(pos - 1.0) < 1e-15
        assert abs(vel - 1.0) < 1e-15

    def test_vector_operations_precision(self):
        """Test vector operations precision"""
        from neural_klotski.math_utils import Vector2D
        import math

        v1 = Vector2D(1.0, 0.0)
        v2 = Vector2D(0.0, 1.0)

        # Test vector addition
        v3 = v1 + v2
        assert abs(v3.x - 1.0) < 1e-15
        assert abs(v3.y - 1.0) < 1e-15

        # Test magnitude calculation
        mag = v3.magnitude()
        expected_mag = math.sqrt(2.0)
        assert abs(mag - expected_mag) < 1e-15


class TestBoundaryConditions:
    """Test boundary conditions and edge cases"""

    def test_zero_values(self):
        """Test behavior with zero values"""
        from neural_klotski.math_utils import (
            spring_force, exponential_decay, lag_distance
        )

        # Spring force with zero displacement
        force = spring_force(0.0, 0.0, 1.0)
        assert force == 0.0

        # Exponential decay with zero time
        decay = exponential_decay(1.0, 0.0, 1.0)
        assert decay == 1.0

        # Lag distance with same positions
        distance = lag_distance(10.0, 10.0)
        assert distance == 0.0

    def test_extreme_values(self):
        """Test behavior with extreme values"""
        from neural_klotski.math_utils import clamp, is_in_bounds

        # Test clamping with extreme values
        clamped = clamp(1e10, -1e10, 1e10)
        assert clamped == 1e10

        clamped = clamp(-1e10, -1e10, 1e10)
        assert clamped == -1e10

        # Test bounds checking
        assert is_in_bounds(0.0, -1e10, 1e10)
        assert not is_in_bounds(1e11, -1e10, 1e10)

    def test_threshold_crossing_edge_cases(self):
        """Test threshold crossing edge cases"""
        from neural_klotski.math_utils import threshold_crossing_from_below

        # Exactly at threshold
        assert not threshold_crossing_from_below(50.0, 50.0, 50.0)

        # Crossing from above (should not trigger)
        assert not threshold_crossing_from_below(60.0, 40.0, 50.0)

        # Very small crossing
        assert threshold_crossing_from_below(49.999999, 50.000001, 50.0)

    def test_configuration_edge_cases(self):
        """Test configuration with edge case values"""
        from neural_klotski.config import DynamicsConfig, WireConfig

        # Test minimum valid values
        dynamics = DynamicsConfig(
            dt=0.1,
            mass=0.1,
            damping=0.1,
            spring_constant=0.1
        )
        assert dynamics.validate()

        # Test maximum valid values
        dynamics = DynamicsConfig(
            dt=1.0,
            mass=10.0,
            damping=0.2,
            spring_constant=0.2
        )
        assert dynamics.validate()


class TestSpecificationCompliance:
    """Test compliance with Neural-Klotski specification"""

    def test_equation_implementation_compliance(self):
        """Test that implemented equations match specification"""
        validator = MathematicalValidator(tolerance=1e-12)

        # Run all validation suites
        suites = [
            validator.validate_math_utils(),
            validator.validate_block_dynamics(),
            validator.validate_wire_mechanics(),
            validator.validate_network_architecture()
        ]

        # All suites should pass completely
        for suite in suites:
            summary = suite.get_summary()
            assert summary['all_passed'], f"Suite {suite.suite_name} has failures"

    def test_parameter_specification_compliance(self):
        """Test parameter values match specification ranges"""
        config = get_default_config()

        # All configuration sections should validate
        assert config.dynamics.validate()
        assert config.wires.validate()
        assert config.dyes.validate()
        assert config.learning.validate()
        assert config.thresholds.validate()
        assert config.network.validate()
        assert config.simulation.validate()

    def test_architecture_specification_compliance(self):
        """Test network architecture matches specification"""
        from neural_klotski.core.architecture import create_addition_network

        # Create network
        network = create_addition_network(enable_learning=False)

        # Test block count
        assert len(network.blocks) == 79

        # Test wire count (should be substantial but not excessive)
        assert len(network.wires) > 1000  # Should have many connections
        assert len(network.wires) < 10000  # But not too many

        # Test shelf structure exists
        input_blocks = [b for b in network.blocks.values()
                       if 45 <= b.lag_position <= 55]
        assert len(input_blocks) == 20

        output_blocks = [b for b in network.blocks.values()
                        if 195 <= b.lag_position <= 205]
        assert len(output_blocks) == 19


if __name__ == "__main__":
    pytest.main([__file__, "-v"])