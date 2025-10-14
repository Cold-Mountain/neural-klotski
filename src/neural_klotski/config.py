"""
Configuration system for Neural-Klotski simulation.

Contains all parameters from Section 9.7 of the specification with bounds checking
and validation functions. All parameters are based on the documented ranges for
stable operation.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import json


@dataclass
class DynamicsConfig:
    """Block dynamics parameters (Section 9.2.1)"""
    # Time integration
    dt: float = 0.5  # Timestep size (0.1-1.0)

    # Physical properties
    mass: float = 1.0  # Block effective mass (typically 1.0)
    damping: float = 0.15  # Damping coefficient (0.1-0.2)
    spring_constant: float = 0.15  # Spring constant k (0.1-0.2)

    # Refractory dynamics
    refractory_kick: float = 20.0  # Velocity kick magnitude (15-25)
    refractory_duration: float = 35.0  # Refractory period timesteps (20-50)

    def validate(self) -> bool:
        """Validate all dynamics parameters are within stable ranges"""
        checks = [
            0.1 <= self.dt <= 1.0,
            self.mass > 0,
            0.1 <= self.damping <= 0.2,
            0.1 <= self.spring_constant <= 0.2,
            15.0 <= self.refractory_kick <= 25.0,
            20.0 <= self.refractory_duration <= 50.0
        ]
        return all(checks)


@dataclass
class WireConfig:
    """Wire/synapse parameters (Section 4)"""
    # Strength parameters
    initial_strength_min: float = 0.8  # Minimum initial strength
    initial_strength_max: float = 2.5  # Maximum initial strength
    strength_min: float = 0.1  # Absolute minimum strength
    strength_max: float = 10.0  # Absolute maximum strength

    # Signal propagation
    signal_speed: float = 100.0  # Signal propagation speed (50-200 lag units/timestep)

    # Fatigue and recovery (Section 9.4.2)
    damage_rate: float = 0.05  # Damage accumulation rate (0.01-0.1)
    repair_rate: float = 0.005  # Repair rate (0.001-0.01)
    max_damage: float = 2.5  # Maximum damage (1.0-5.0)
    fatigue_rate: float = 0.01  # Fatigue accumulation rate (0.005-0.02)

    def validate(self) -> bool:
        """Validate wire parameters"""
        checks = [
            self.initial_strength_min >= 0,
            self.initial_strength_max > self.initial_strength_min,
            self.strength_min >= 0,
            self.strength_max > self.strength_min,
            50.0 <= self.signal_speed <= 200.0,
            0.01 <= self.damage_rate <= 0.1,
            0.001 <= self.repair_rate <= 0.01,
            1.0 <= self.max_damage <= 5.0,
            0.005 <= self.fatigue_rate <= 0.02
        ]
        return all(checks)


@dataclass
class DyeConfig:
    """Dye dynamics parameters (Section 9.3)"""
    # Diffusion and decay
    diffusion_coefficient: float = 1.0  # Diffusion coefficient D (0.5-2.0)
    decay_time_constant: float = 500.0  # Decay time constant τ (100-1000 timesteps)

    # Enhancement
    enhancement_factor: float = 2.5  # Enhancement factor α (1.0-3.0)
    injection_amount: float = 0.7  # Injection concentration (0.5-1.0)

    # Temporal eligibility
    eligibility_window: float = 100.0  # Temporal window for eligibility (50-200 timesteps)

    def validate(self) -> bool:
        """Validate dye parameters"""
        checks = [
            0.5 <= self.diffusion_coefficient <= 2.0,
            100.0 <= self.decay_time_constant <= 1000.0,
            1.0 <= self.enhancement_factor <= 3.0,
            0.5 <= self.injection_amount <= 1.0,
            50.0 <= self.eligibility_window <= 200.0
        ]
        return all(checks)


@dataclass
class LearningConfig:
    """Learning and plasticity parameters (Section 9.4)"""
    # Hebbian learning
    wire_learning_rate: float = 0.005  # Wire strength learning rate η_w (0.001-0.01)
    dye_amplification: float = 3.5  # Dye amplification factor β (2.0-5.0)

    # STDP
    stdp_time_constant: float = 15.0  # STDP time constant τ_STDP (10-20 timesteps)
    temporal_window: float = 30.0  # Temporal correlation window (10-50 timesteps)

    # Threshold adaptation
    threshold_learning_rate: float = 0.005  # Threshold adaptation rate η_τ (0.001-0.01)
    target_firing_rate: float = 0.1  # Target firing rate (0.05-0.2)
    measurement_window: float = 5000.0  # Firing rate measurement window (1000-10000 timesteps)

    def validate(self) -> bool:
        """Validate learning parameters"""
        checks = [
            0.001 <= self.wire_learning_rate <= 0.01,
            2.0 <= self.dye_amplification <= 5.0,
            10.0 <= self.stdp_time_constant <= 20.0,
            10.0 <= self.temporal_window <= 50.0,
            0.001 <= self.threshold_learning_rate <= 0.01,
            0.05 <= self.target_firing_rate <= 0.2,
            1000.0 <= self.measurement_window <= 10000.0
        ]
        return all(checks)


@dataclass
class ThresholdConfig:
    """Block threshold parameters (Section 3.1)"""
    # Threshold ranges for different shelf types
    input_threshold_min: float = 40.0
    input_threshold_max: float = 50.0
    hidden_threshold_min: float = 45.0
    hidden_threshold_max: float = 60.0
    output_threshold_min: float = 50.0
    output_threshold_max: float = 60.0

    # Absolute bounds
    threshold_min: float = 30.0  # Absolute minimum threshold
    threshold_max: float = 80.0  # Absolute maximum threshold

    def validate(self) -> bool:
        """Validate threshold parameters"""
        checks = [
            self.threshold_min <= self.input_threshold_min < self.input_threshold_max,
            self.threshold_min <= self.hidden_threshold_min < self.hidden_threshold_max,
            self.threshold_min <= self.output_threshold_min < self.output_threshold_max,
            self.input_threshold_max <= self.threshold_max,
            self.hidden_threshold_max <= self.threshold_max,
            self.output_threshold_max <= self.threshold_max
        ]
        return all(checks)


@dataclass
class NetworkConfig:
    """Network architecture parameters (Section 8.2)"""
    # Addition network architecture
    total_blocks: int = 79
    input_blocks: int = 20
    hidden1_blocks: int = 20
    hidden2_blocks: int = 20
    output_blocks: int = 19

    # Shelf lag positions (centers)
    shelf1_lag_center: float = 50.0  # Input shelf (45-55)
    shelf2_lag_center: float = 100.0  # Hidden 1 shelf (95-105)
    shelf3_lag_center: float = 150.0  # Hidden 2 shelf (145-155)
    shelf4_lag_center: float = 200.0  # Output shelf (195-205)
    shelf_width: float = 10.0  # Half-width of each shelf

    # Connectivity parameters
    local_connections: int = 20  # K nearest neighbors
    longrange_connections: int = 5  # L long-range connections
    longrange_min_distance: float = 50.0  # Minimum distance for long-range

    # Color distribution in hidden layers
    hidden_red_fraction: float = 0.7  # 70% red blocks
    hidden_blue_fraction: float = 0.25  # 25% blue blocks
    hidden_yellow_fraction: float = 0.05  # 5% yellow blocks

    def validate(self) -> bool:
        """Validate network architecture parameters"""
        checks = [
            self.input_blocks + self.hidden1_blocks + self.hidden2_blocks + self.output_blocks == self.total_blocks,
            self.local_connections > 0,
            self.longrange_connections >= 0,
            self.longrange_min_distance > 0,
            abs(self.hidden_red_fraction + self.hidden_blue_fraction + self.hidden_yellow_fraction - 1.0) < 1e-6
        ]
        return all(checks)


@dataclass
class SimulationConfig:
    """Simulation control parameters"""
    # Training parameters
    max_epochs: int = 5000
    success_threshold: float = 0.95  # 95% accuracy required
    computation_timesteps: int = 500  # T_compute duration

    # Input/output scaling
    input_scaling_factor: float = 100.0  # Scale factor for input positioning
    output_threshold: float = 55.0  # Threshold for output decoding

    def validate(self) -> bool:
        """Validate simulation parameters"""
        checks = [
            self.max_epochs > 0,
            0.0 <= self.success_threshold <= 1.0,
            self.computation_timesteps > 0,
            self.input_scaling_factor > 0,
            self.output_threshold > 0
        ]
        return all(checks)


@dataclass
class NeuralKlotskiConfig:
    """Complete configuration for Neural-Klotski system"""
    dynamics: DynamicsConfig = None
    wires: WireConfig = None
    dyes: DyeConfig = None
    learning: LearningConfig = None
    thresholds: ThresholdConfig = None
    network: NetworkConfig = None
    simulation: SimulationConfig = None

    def __post_init__(self):
        """Initialize with defaults if None"""
        if self.dynamics is None:
            self.dynamics = DynamicsConfig()
        if self.wires is None:
            self.wires = WireConfig()
        if self.dyes is None:
            self.dyes = DyeConfig()
        if self.learning is None:
            self.learning = LearningConfig()
        if self.thresholds is None:
            self.thresholds = ThresholdConfig()
        if self.network is None:
            self.network = NetworkConfig()
        if self.simulation is None:
            self.simulation = SimulationConfig()

    def validate(self) -> bool:
        """Validate all configuration components"""
        return all([
            self.dynamics.validate(),
            self.wires.validate(),
            self.dyes.validate(),
            self.learning.validate(),
            self.thresholds.validate(),
            self.network.validate(),
            self.simulation.validate()
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            'dynamics': self.dynamics.__dict__,
            'wires': self.wires.__dict__,
            'dyes': self.dyes.__dict__,
            'learning': self.learning.__dict__,
            'thresholds': self.thresholds.__dict__,
            'network': self.network.__dict__,
            'simulation': self.simulation.__dict__
        }

    def save(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'NeuralKlotskiConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls(
            dynamics=DynamicsConfig(**data['dynamics']),
            wires=WireConfig(**data['wires']),
            dyes=DyeConfig(**data['dyes']),
            learning=LearningConfig(**data['learning']),
            thresholds=ThresholdConfig(**data['thresholds']),
            network=NetworkConfig(**data['network']),
            simulation=SimulationConfig(**data['simulation'])
        )


def get_default_config() -> NeuralKlotskiConfig:
    """Get default configuration with parameters optimized for addition task"""
    return NeuralKlotskiConfig()


def validate_parameter_bounds(value: float, bounds: Tuple[float, float], name: str) -> bool:
    """Utility function to validate a parameter is within specified bounds"""
    min_val, max_val = bounds
    if not (min_val <= value <= max_val):
        raise ValueError(f"Parameter {name} = {value} is outside valid range [{min_val}, {max_val}]")
    return True


if __name__ == "__main__":
    # Test configuration validation
    config = get_default_config()
    print(f"Default configuration is valid: {config.validate()}")

    # Save default configuration
    config.save("default_config.json")
    print("Default configuration saved to default_config.json")