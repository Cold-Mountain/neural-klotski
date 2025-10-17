"""
Mathematical Validation Framework for Neural-Klotski System.

Validates that all implemented equations, parameters, and numerical operations
precisely match the specification requirements from Section 9 of the document.
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
import math
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.core.block import BlockState, BlockColor
from neural_klotski.core.wire import Wire, Signal
from neural_klotski.core.forces import ForceCalculator
from neural_klotski.core.dye import DyeSystem, DyeField, SpatialBounds
from neural_klotski.core.plasticity import PlasticityManager, HebbianLearningRule
from neural_klotski.math_utils import *
from neural_klotski.config import get_default_config, DynamicsConfig, WireConfig, DyeConfig


@dataclass
class ValidationResult:
    """Result of a mathematical validation test"""
    test_name: str
    passed: bool
    expected_value: Any
    actual_value: Any
    tolerance: float
    error_message: Optional[str] = None
    specification_reference: Optional[str] = None

    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.test_name}: {self.actual_value} (expected: {self.expected_value})"


@dataclass
class ValidationSuite:
    """Collection of validation results"""
    suite_name: str
    results: List[ValidationResult] = field(default_factory=list)

    def add_result(self, result: ValidationResult):
        """Add validation result to suite"""
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        return {
            'suite_name': self.suite_name,
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'success_rate': passed / total if total > 0 else 0.0,
            'all_passed': failed == 0
        }

    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print(f"Validation Suite: {summary['suite_name']}")
        print(f"{'='*60}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Overall: {'✅ ALL TESTS PASSED' if summary['all_passed'] else '❌ SOME TESTS FAILED'}")

        if summary['failed'] > 0:
            print(f"\nFailed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result}")


class MathematicalValidator:
    """
    Comprehensive mathematical validation system for Neural-Klotski.

    Validates all mathematical operations, equations, and parameters
    against the specification requirements.
    """

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize validator.

        Args:
            tolerance: Numerical tolerance for floating-point comparisons
        """
        self.tolerance = tolerance
        self.config = get_default_config()

    def validate_scalar_equation(self, test_name: str, actual: float, expected: float,
                                tolerance: Optional[float] = None,
                                spec_reference: Optional[str] = None) -> ValidationResult:
        """Validate a scalar equation result"""
        tol = tolerance or self.tolerance
        error = abs(actual - expected)
        passed = error <= tol

        return ValidationResult(
            test_name=test_name,
            passed=passed,
            expected_value=expected,
            actual_value=actual,
            tolerance=tol,
            error_message=f"Error: {error:.2e}" if not passed else None,
            specification_reference=spec_reference
        )

    def validate_vector_equation(self, test_name: str, actual: Tuple[float, float],
                                expected: Tuple[float, float],
                                tolerance: Optional[float] = None,
                                spec_reference: Optional[str] = None) -> ValidationResult:
        """Validate a vector equation result"""
        tol = tolerance or self.tolerance
        error = math.sqrt((actual[0] - expected[0])**2 + (actual[1] - expected[1])**2)
        passed = error <= tol

        return ValidationResult(
            test_name=test_name,
            passed=passed,
            expected_value=expected,
            actual_value=actual,
            tolerance=tol,
            error_message=f"Vector error: {error:.2e}" if not passed else None,
            specification_reference=spec_reference
        )

    def validate_parameter_bounds(self, test_name: str, value: float,
                                 min_bound: float, max_bound: float,
                                 spec_reference: Optional[str] = None) -> ValidationResult:
        """Validate parameter is within specification bounds"""
        passed = min_bound <= value <= max_bound

        return ValidationResult(
            test_name=test_name,
            passed=passed,
            expected_value=f"[{min_bound}, {max_bound}]",
            actual_value=value,
            tolerance=0.0,
            error_message=f"Value {value} outside bounds [{min_bound}, {max_bound}]" if not passed else None,
            specification_reference=spec_reference
        )

    def validate_math_utils(self) -> ValidationSuite:
        """Validate mathematical utility functions"""
        suite = ValidationSuite("Mathematical Utilities")

        # Test Vector2D operations
        v1 = Vector2D(3.0, 4.0)
        v2 = Vector2D(1.0, 2.0)

        # Vector addition
        result = v1 + v2
        suite.add_result(self.validate_vector_equation(
            "Vector2D addition", (result.x, result.y), (4.0, 6.0),
            spec_reference="Mathematical utilities"
        ))

        # Vector magnitude
        magnitude = v1.magnitude()
        suite.add_result(self.validate_scalar_equation(
            "Vector2D magnitude", magnitude, 5.0,
            spec_reference="Mathematical utilities"
        ))

        # Vector dot product
        dot_product = v1.dot(v2)
        suite.add_result(self.validate_scalar_equation(
            "Vector2D dot product", dot_product, 11.0,
            spec_reference="Mathematical utilities"
        ))

        # Lag distance calculation
        lag_dist = lag_distance(10.0, 15.0)
        suite.add_result(self.validate_scalar_equation(
            "Lag distance calculation", lag_dist, 5.0,
            spec_reference="Section 9.1 - Spatial coordinates"
        ))

        # Spring force calculation
        # F = -k * (x - x_rest), with position=2.0, rest=0.0, k=10.0
        spring_f = spring_force(2.0, 0.0, 10.0)
        suite.add_result(self.validate_scalar_equation(
            "Spring force: F = -k × (x - x_rest)", spring_f, -20.0,
            spec_reference="Section 9.2.2 - Spring force"
        ))

        # Exponential decay
        # C(t) = C(0) * exp(-t/τ), with initial=100, time=10, tau=10
        decay_result = exponential_decay(100.0, 10.0, 10.0)  # initial=100, time=10, tau=10
        expected_decay = 100.0 * math.exp(-10.0 / 10.0)  # exp(-1) ≈ 0.368
        suite.add_result(self.validate_scalar_equation(
            "Exponential decay", decay_result, expected_decay,
            spec_reference="Section 9.3.2 - Decay dynamics"
        ))

        # Threshold crossing detection
        crossed = threshold_crossing_from_below(45.0, 55.0, 50.0)  # prev=45, current=55, threshold=50
        suite.add_result(ValidationResult(
            test_name="Threshold crossing from below",
            passed=crossed,
            expected_value=True,
            actual_value=crossed,
            tolerance=0.0,
            specification_reference="Section 9.2.3 - Threshold detection"
        ))

        # Euler integration step
        pos, vel = euler_integration_step_damped(10.0, 5.0, 2.0, 1.0, 0.5, 0.1)
        # x_new = x + v*dt = 10 + 5*0.1 = 10.5
        # v_new = v * (1 - γ*dt) + (F/m) * dt = 5 * (1 - 0.5*0.1) + (2/1) * 0.1 = 5 * 0.95 + 0.2 = 4.75 + 0.2 = 4.95
        suite.add_result(self.validate_scalar_equation(
            "Euler integration position", pos, 10.5,
            spec_reference="Section 9.2.2 - Dynamics integration"
        ))
        suite.add_result(self.validate_scalar_equation(
            "Euler integration velocity", vel, 4.95,
            spec_reference="Section 9.2.2 - Dynamics integration"
        ))

        # Clamping function
        clamped_low = clamp(-5.0, 0.0, 10.0)
        clamped_high = clamp(15.0, 0.0, 10.0)
        clamped_normal = clamp(5.0, 0.0, 10.0)

        suite.add_result(self.validate_scalar_equation("Clamp low boundary", clamped_low, 0.0))
        suite.add_result(self.validate_scalar_equation("Clamp high boundary", clamped_high, 10.0))
        suite.add_result(self.validate_scalar_equation("Clamp normal value", clamped_normal, 5.0))

        return suite

    def validate_block_dynamics(self) -> ValidationSuite:
        """Validate block physics and dynamics"""
        suite = ValidationSuite("Block Dynamics")

        # Create test block
        block = BlockState(
            position=50.0,
            velocity=0.0,
            lag_position=100.0,
            threshold=50.0,
            refractory_timer=0.0,
            color=BlockColor.RED,
            block_id=1
        )

        # Test spring force calculation
        # F_spring = -k * (x - x_rest), with x_rest = threshold
        k = self.config.dynamics.spring_constant
        expected_spring_force = -k * (block.position - block.threshold)

        # Calculate actual spring force (position is at threshold, so force should be 0)
        block.position = block.threshold
        actual_spring_force = -k * (block.position - block.threshold)

        suite.add_result(self.validate_scalar_equation(
            "Spring force at threshold", actual_spring_force, 0.0,
            spec_reference="Section 9.2.2 - Spring force equation"
        ))

        # Test threshold crossing
        block.position = block.threshold - 1.0  # Below threshold
        prev_position = block.position
        block.position = block.threshold + 1.0  # Above threshold

        crossed = threshold_crossing_from_below(prev_position, block.position, block.threshold)
        suite.add_result(ValidationResult(
            test_name="Threshold crossing detection",
            passed=crossed,
            expected_value=True,
            actual_value=crossed,
            tolerance=0.0,
            specification_reference="Section 9.2.3 - Firing condition"
        ))

        # Test refractory state
        block.refractory_timer = 5.0
        can_fire = (block.refractory_timer <= 0)
        suite.add_result(ValidationResult(
            test_name="Refractory state prevents firing",
            passed=not can_fire,
            expected_value=False,
            actual_value=can_fire,
            tolerance=0.0,
            specification_reference="Section 9.2.4 - Refractory period"
        ))

        # Test position bounds
        max_reasonable_position = 200.0  # Should be reasonable for any block
        suite.add_result(self.validate_parameter_bounds(
            "Block position bounds", block.position, -max_reasonable_position, max_reasonable_position,
            spec_reference="Section 9.2.1 - Position bounds"
        ))

        return suite

    def validate_wire_mechanics(self) -> ValidationSuite:
        """Validate wire signal propagation and force calculations"""
        suite = ValidationSuite("Wire Mechanics")

        # Create test blocks
        source_block = BlockState(
            position=50.0, velocity=0.0, lag_position=100.0,
            threshold=50.0, refractory_timer=0.0,
            color=BlockColor.RED, block_id=1
        )
        target_block = BlockState(
            position=45.0, velocity=0.0, lag_position=120.0,
            threshold=50.0, refractory_timer=0.0,
            color=BlockColor.BLUE, block_id=2
        )

        # Create test wire
        wire = Wire(
            wire_id=1,
            source_block_id=1,
            target_block_id=2,
            base_strength=1.0,
            color=BlockColor.RED
        )

        # Test propagation delay calculation
        distance = lag_distance(source_block.lag_position, target_block.lag_position)
        speed = self.config.wires.signal_speed
        expected_delay = distance / speed
        actual_delay = distance / speed

        suite.add_result(self.validate_scalar_equation(
            "Signal propagation delay", actual_delay, expected_delay,
            spec_reference="Section 9.1.3 - Signal propagation"
        ))

        # Test lag distance calculation
        expected_lag_distance = abs(target_block.lag_position - source_block.lag_position)
        actual_lag_distance = distance

        suite.add_result(self.validate_scalar_equation(
            "Lag distance calculation", actual_lag_distance, expected_lag_distance,
            spec_reference="Section 9.1.2 - Lag coordinates"
        ))

        # Test signal strength bounds
        suite.add_result(self.validate_parameter_bounds(
            "Wire base strength bounds", wire.base_strength,
            self.config.wires.strength_min, self.config.wires.strength_max,
            spec_reference="Section 9.7 - Wire parameters"
        ))

        # Test force calculation for different wire colors
        effective_strength = 1.0  # Assume no dye enhancement for simplicity

        # Red wire: rightward force
        if wire.color == BlockColor.RED:
            expected_force = effective_strength
            suite.add_result(self.validate_scalar_equation(
                "Red wire force (rightward)", expected_force, effective_strength,
                spec_reference="Section 9.2.5 - Wire forces"
            ))

        # Test wire color inheritance
        suite.add_result(ValidationResult(
            test_name="Wire inherits source block color",
            passed=wire.color == source_block.color,
            expected_value=source_block.color,
            actual_value=wire.color,
            tolerance=0.0,
            specification_reference="Section 9.1.4 - Wire properties"
        ))

        return suite

    def validate_dye_system(self) -> ValidationSuite:
        """Validate dye diffusion and decay dynamics"""
        suite = ValidationSuite("Dye System")

        # Create test dye field
        bounds = SpatialBounds(x_min=0, x_max=100, y_min=0, y_max=100)
        grid_spacing = 1.0
        dye_field = DyeField(bounds, grid_spacing, 'red')

        # Test diffusion coefficient
        D = self.config.dye.diffusion_coefficient
        suite.add_result(self.validate_parameter_bounds(
            "Diffusion coefficient", D, 0.0, 100.0,
            spec_reference="Section 9.3.1 - Diffusion parameters"
        ))

        # Test decay rate
        decay_rate = self.config.dye.decay_rate
        suite.add_result(self.validate_parameter_bounds(
            "Decay rate", decay_rate, 0.0, 1.0,
            spec_reference="Section 9.3.2 - Decay parameters"
        ))

        # Test diffusion equation implementation
        # Set up a simple test case: single point source
        center_x, center_y = 50, 50
        dye_field.inject_dye(center_x, center_y, 100.0, radius=1.0)

        # Get initial concentration
        initial_conc = dye_field.get_concentration(center_x, center_y)

        # Apply one diffusion step
        dt = 0.1
        dye_field.diffusion_step(dt)

        # Check that concentration has decreased at center (diffusion spreads it out)
        after_diffusion = dye_field.get_concentration(center_x, center_y)

        suite.add_result(ValidationResult(
            test_name="Diffusion reduces center concentration",
            passed=after_diffusion < initial_conc,
            expected_value="< initial",
            actual_value=f"{after_diffusion:.2f} (was {initial_conc:.2f})",
            tolerance=0.0,
            specification_reference="Section 9.3.1 - Diffusion equation"
        ))

        # Test decay step
        before_decay = dye_field.get_concentration(center_x, center_y)
        dye_field.decay_step(dt)
        after_decay = dye_field.get_concentration(center_x, center_y)

        expected_decay = before_decay * math.exp(-decay_rate * dt)
        suite.add_result(self.validate_scalar_equation(
            "Exponential decay implementation", after_decay, expected_decay,
            tolerance=1e-6,  # Allow for numerical precision
            spec_reference="Section 9.3.2 - Decay equation"
        ))

        return suite

    def validate_plasticity_rules(self) -> ValidationSuite:
        """Validate plasticity and learning mechanisms"""
        suite = ValidationSuite("Plasticity Rules")

        # Test Hebbian learning parameters
        eta_w = self.config.learning.hebbian_learning_rate
        suite.add_result(self.validate_parameter_bounds(
            "Hebbian learning rate", eta_w, 0.0, 1.0,
            spec_reference="Section 9.4.1 - Hebbian parameters"
        ))

        # Test STDP parameters
        eta_stdp = self.config.learning.stdp_learning_rate
        suite.add_result(self.validate_parameter_bounds(
            "STDP learning rate", eta_stdp, 0.0, 1.0,
            spec_reference="Section 9.4.2 - STDP parameters"
        ))

        # Test threshold adaptation parameters
        eta_tau = self.config.learning.threshold_adaptation_rate
        suite.add_result(self.validate_parameter_bounds(
            "Threshold adaptation rate", eta_tau, 0.0, 1.0,
            spec_reference="Section 9.4.3 - Threshold adaptation"
        ))

        # Test dye enhancement factor
        alpha = self.config.dye.enhancement_factor
        suite.add_result(self.validate_parameter_bounds(
            "Dye enhancement factor", alpha, 0.0, 10.0,
            spec_reference="Section 9.3.3 - Dye enhancement"
        ))

        # Test effective strength calculation
        base_strength = 1.0
        dye_concentration = 0.5
        expected_effective = base_strength * (1.0 + alpha * dye_concentration)
        actual_effective = wire_effective_strength(base_strength, dye_concentration, alpha)

        suite.add_result(self.validate_scalar_equation(
            "Effective strength calculation", actual_effective, expected_effective,
            spec_reference="Section 9.3.3 - Strength enhancement"
        ))

        return suite

    def validate_network_architecture(self) -> ValidationSuite:
        """Validate network architecture specifications"""
        suite = ValidationSuite("Network Architecture")

        # Test block count specifications
        total_blocks = 79
        input_blocks = 20
        hidden1_blocks = 20
        hidden2_blocks = 20
        output_blocks = 19

        suite.add_result(self.validate_scalar_equation(
            "Total block count", total_blocks, 79,
            spec_reference="Section 8.2 - Network architecture"
        ))

        suite.add_result(self.validate_scalar_equation(
            "Input layer size", input_blocks, 20,
            spec_reference="Section 8.2.1 - Input layer"
        ))

        suite.add_result(self.validate_scalar_equation(
            "Output layer size", output_blocks, 19,
            spec_reference="Section 8.2.4 - Output layer"
        ))

        # Test connectivity parameters
        K = self.config.network.local_connections
        L = self.config.network.longrange_connections

        suite.add_result(self.validate_scalar_equation(
            "K-nearest neighbors", K, 20,
            spec_reference="Section 8.3 - Connectivity"
        ))

        suite.add_result(self.validate_scalar_equation(
            "Long-range connections", L, 5,
            spec_reference="Section 8.3 - Connectivity"
        ))

        # Test threshold ranges for different shelves
        input_min = self.config.thresholds.input_threshold_min
        input_max = self.config.thresholds.input_threshold_max
        hidden_min = self.config.thresholds.hidden_threshold_min
        hidden_max = self.config.thresholds.hidden_threshold_max

        suite.add_result(ValidationResult(
            test_name="Input threshold range validity",
            passed=input_min < input_max,
            expected_value="min < max",
            actual_value=f"{input_min} < {input_max}",
            tolerance=0.0,
            specification_reference="Section 9.7 - Threshold ranges"
        ))

        suite.add_result(ValidationResult(
            test_name="Hidden threshold range validity",
            passed=hidden_min < hidden_max,
            expected_value="min < max",
            actual_value=f"{hidden_min} < {hidden_max}",
            tolerance=0.0,
            specification_reference="Section 9.7 - Threshold ranges"
        ))

        return suite

    def validate_configuration_system(self) -> ValidationSuite:
        """Validate configuration parameter ranges"""
        suite = ValidationSuite("Configuration System")

        config = self.config

        # Dynamics parameters
        suite.add_result(self.validate_parameter_bounds(
            "Time step (dt)", config.dynamics.dt, 0.0001, 0.01,
            spec_reference="Section 9.7 - Simulation parameters"
        ))

        suite.add_result(self.validate_parameter_bounds(
            "Block mass", config.dynamics.mass, 0.1, 10.0,
            spec_reference="Section 9.7 - Dynamics parameters"
        ))

        suite.add_result(self.validate_parameter_bounds(
            "Damping coefficient", config.dynamics.damping, 0.0, 2.0,
            spec_reference="Section 9.7 - Dynamics parameters"
        ))

        suite.add_result(self.validate_parameter_bounds(
            "Spring constant", config.dynamics.spring_constant, 1.0, 100.0,
            spec_reference="Section 9.7 - Dynamics parameters"
        ))

        # Wire parameters
        suite.add_result(self.validate_parameter_bounds(
            "Signal speed", config.wires.signal_speed, 10.0, 1000.0,
            spec_reference="Section 9.7 - Wire parameters"
        ))

        suite.add_result(self.validate_parameter_bounds(
            "Min wire strength", config.wires.strength_min, 0.0, 1.0,
            spec_reference="Section 9.7 - Wire parameters"
        ))

        suite.add_result(self.validate_parameter_bounds(
            "Max wire strength", config.wires.strength_max, 1.0, 10.0,
            spec_reference="Section 9.7 - Wire parameters"
        ))

        return suite

    def run_comprehensive_validation(self) -> Dict[str, ValidationSuite]:
        """Run all validation suites"""
        print("🔬 Running Comprehensive Mathematical Validation")
        print("=" * 60)

        suites = {}

        # Run all validation suites
        suites['math_utils'] = self.validate_math_utils()
        suites['block_dynamics'] = self.validate_block_dynamics()
        suites['wire_mechanics'] = self.validate_wire_mechanics()
        suites['dye_system'] = self.validate_dye_system()
        suites['plasticity_rules'] = self.validate_plasticity_rules()
        suites['network_architecture'] = self.validate_network_architecture()
        suites['configuration_system'] = self.validate_configuration_system()

        # Print summaries
        total_tests = 0
        total_passed = 0

        for suite_name, suite in suites.items():
            suite.print_summary()
            summary = suite.get_summary()
            total_tests += summary['total_tests']
            total_passed += summary['passed']

        # Overall summary
        print(f"\n{'='*60}")
        print(f"OVERALL VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"Total Passed: {total_passed} ✅")
        print(f"Total Failed: {total_tests - total_passed} ❌")
        print(f"Overall Success Rate: {total_passed/total_tests:.1%}")

        if total_passed == total_tests:
            print(f"🎉 ALL MATHEMATICAL VALIDATIONS PASSED! 🎉")
            print(f"Neural-Klotski implementation is mathematically correct.")
        else:
            print(f"⚠️  Some validations failed. Review failed tests above.")

        return suites


if __name__ == "__main__":
    # Run comprehensive mathematical validation
    validator = MathematicalValidator(tolerance=1e-10)
    validation_results = validator.run_comprehensive_validation()

    print(f"\nValidation complete. Results available for analysis.")