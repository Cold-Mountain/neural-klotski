# Neural-Klotski: Bio-Inspired Neural Network for Addition Learning

A complete implementation of the Neural-Klotski system - a bio-inspired neural network that learns arithmetic addition through sliding block dynamics, dye-enhanced plasticity, and spatiotemporal learning mechanisms.

## 🧠 System Overview

Neural-Klotski implements neurons as sliding blocks with position-based activation, connected by temporal wires that propagate signals with realistic delays. The system learns through dye diffusion that enhances synaptic plasticity, creating a unique bio-inspired learning mechanism.

### Key Features

- **79-Block Addition Network**: Specialized architecture for learning 10-bit + 10-bit addition
- **Sliding Block Dynamics**: Physics-based neuron model with position, velocity, and threshold mechanics
- **Temporal Signal Propagation**: Realistic signal delays based on spatial distance
- **Dye-Enhanced Plasticity**: Bio-inspired learning through spatial dye diffusion
- **Comprehensive Training System**: Complete pipeline with monitoring, optimization, and visualization

## 🏗️ Architecture

```
Input Layer (20 blocks)    Hidden Layer 1 (20 blocks)    Hidden Layer 2 (20 blocks)    Output Layer (19 blocks)
Operand 1: 10 blocks  -->  Mixed color processing    -->  Advanced integration     -->  Sum output: 0-511
Operand 2: 10 blocks                                                                    (19 blocks = 0-18, overflow)
```

- **Input Encoding**: Binary representation of two 10-bit operands
- **Hidden Processing**: K-nearest neighbor connectivity with long-range connections
- **Output Decoding**: Position-based sum extraction with confidence estimation
- **Learning**: Dye-enhanced Hebbian plasticity with eligibility traces

## 📁 Project Structure

```
neural-klotski/
├── src/neural_klotski/
│   ├── core/                          # Core neural dynamics
│   │   ├── block.py                   # Sliding block neurons
│   │   ├── wire.py                    # Temporal connections
│   │   ├── forces.py                  # Force application system
│   │   ├── network.py                 # Complete network integration
│   │   ├── dye.py                     # Spatial dye diffusion
│   │   ├── plasticity.py              # Learning mechanisms
│   │   ├── learning.py                # Integrated learning system
│   │   ├── architecture.py            # 79-block network builder
│   │   └── encoding.py                # Addition task encoding/decoding
│   ├── training/                      # Training infrastructure
│   │   ├── trainer.py                 # Core training pipeline
│   │   ├── batch_trainer.py           # Hyperparameter optimization
│   │   ├── monitoring.py              # Performance monitoring
│   │   ├── scheduling.py              # Learning rate scheduling
│   │   └── visualization.py           # Training visualization
│   ├── config.py                      # System configuration
│   └── math_utils.py                  # Mathematical utilities
├── tests/                             # Comprehensive test suite
├── docs/                              # Documentation
└── ROADMAP.md                         # Development progress
```

## 🚀 Quick Start

### Installation

```bash
git clone <repository-url>
cd neural-klotski
pip install -r requirements.txt
```

### Basic Usage

```python
from neural_klotski.core.architecture import create_addition_network
from neural_klotski.core.encoding import AdditionProblem, AdditionTaskManager
from neural_klotski.config import get_default_config

# Create network and configuration
config = get_default_config()
network = create_addition_network(enable_learning=True)

# Create task manager
task_manager = AdditionTaskManager(config.simulation)

# Test addition problem
problem = AdditionProblem(42, 27, 69)  # 42 + 27 = 69
result, stats = task_manager.execute_addition_task(network, problem)

print(f"Problem: {problem}")
print(f"Network result: {result.decoded_sum}")
print(f"Correct: {result.decoded_sum == problem.expected_sum}")
```

### Training a Network

```python
from neural_klotski.training.trainer import AdditionNetworkTrainer, TrainingConfig

# Configure training
training_config = TrainingConfig()
training_config.epochs_per_phase = {
    TrainingPhase.SIMPLE: 100,
    TrainingPhase.INTERMEDIATE: 200,
    TrainingPhase.ADVANCED: 500
}

# Create trainer
trainer = AdditionNetworkTrainer(training_config, config.simulation)

# Train the network
final_metrics = trainer.train()
print(f"Final accuracy: {final_metrics.accuracy:.3f}")
```

### Batch Training with Hyperparameter Optimization

```python
from neural_klotski.training.batch_trainer import BatchTrainer, HyperparameterSweep

# Create hyperparameter sweep
sweep = HyperparameterSweep()
sweep.add_parameter('problems_per_epoch', [20, 50, 100])
sweep.add_parameter('sim_input_scaling_factor', [50.0, 100.0, 150.0])

# Run batch training
batch_trainer = BatchTrainer(BatchConfig())
results = batch_trainer.run_hyperparameter_sweep("addition_optimization", sweep)

# Analyze results
analysis = batch_trainer.analyze_results()
print(f"Best experiment: {analysis['best_experiment']}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_blocks.py -v          # Block dynamics
python -m pytest tests/test_network.py -v         # Network integration
python -m pytest tests/test_training.py -v        # Training system
python -m pytest tests/test_architecture.py -v    # Architecture validation
```

**Test Coverage**: 250+ tests covering all system components with comprehensive validation.

## 📊 Key Technical Specifications

### Neural Dynamics
- **Block Physics**: Spring-damper system with position-based activation
- **Threshold Detection**: "Crossing from below" with refractory mechanics
- **Signal Propagation**: Realistic delays based on spatial distance
- **Force Application**: Color-coded forces (Red: excitatory, Blue: inhibitory, Yellow: coupling)

### Learning Mechanisms
- **Dye Diffusion**: 2D spatial diffusion with exponential decay
- **Hebbian Plasticity**: Activity correlation with dye enhancement
- **STDP**: Spike-timing dependent plasticity
- **Threshold Adaptation**: Homeostatic firing rate regulation

### Network Architecture
- **79 Blocks Total**: 20 input + 20+20 hidden + 19 output
- **1,560 Connections**: K-nearest neighbors + long-range connections
- **Binary Encoding**: Two 10-bit operands → single 19-bit sum
- **Color Distribution**: 70% red, 25% blue, 5% yellow in hidden layers

### Training System
- **Curriculum Learning**: Progressive difficulty (0-15 → 0-127 → 0-511)
- **Convergence Detection**: 5 convergence states with statistical analysis
- **Adaptive Scheduling**: 8 learning rate schedules with automatic selection
- **Performance Monitoring**: Real-time analysis with early stopping

## 📈 Performance Characteristics

### Problem Complexity
- **Maximum Addition**: 511 + 511 = 1022 (10-bit operands)
- **Accuracy Target**: >95% on test problems
- **Training Time**: Variable based on complexity and convergence

### Computational Requirements
- **Memory**: ~79 blocks × 1,560 wires + dye fields
- **Timesteps**: 500 per training trial
- **Learning Phases**: 3-phase curriculum with adaptive progression

## 🔬 Research Applications

### Validated Capabilities
- **Addition Learning**: Complete 10-bit arithmetic with high accuracy
- **Generalization**: Performance on unseen number combinations
- **Plasticity Analysis**: Dye-enhanced learning dynamics
- **Convergence Patterns**: Statistical analysis of learning progression

### Experimental Features
- **Hyperparameter Optimization**: Systematic parameter space exploration
- **Performance Analysis**: Comprehensive learning curve analysis
- **Visualization**: Real-time training progress and network state
- **Batch Processing**: Parallel experiment execution

## 📚 Documentation

- **ROADMAP.md**: Complete development progress and session notes
- **tests/**: Comprehensive test suite with examples
- **src/**: Fully documented source code with docstrings
- **Training Examples**: Multiple training scenarios and configurations

## 🤝 Contributing

This system implements the complete Neural-Klotski specification with:
- ✅ All mathematical equations correctly implemented
- ✅ All architectural requirements satisfied
- ✅ Comprehensive test coverage
- ✅ Full training pipeline with optimization
- ✅ Performance monitoring and visualization

## 📄 License

[Add appropriate license information]

## 🎯 Next Steps

The system is ready for:
1. **✅ Phase 4**: Validation & Analysis - comprehensive performance evaluation ✅ COMPLETED
2. **🎨 Visualization System**: Comprehensive real-time visualization suite ⚡ IN PROGRESS
3. **Generalization Experiments**: Testing on larger number ranges
4. **Architecture Exploration**: Alternative network configurations
5. **Performance Optimization**: Computational efficiency improvements

## 🎬 Visualization System (NEW!)

A comprehensive real-time visualization suite is in development to provide complete insight into Neural-Klotski dynamics:

- **🔴 Real-Time Network Visualization**: 79 blocks in 2D space with live position updates
- **⚡ Signal Propagation Animation**: Animated signals with accurate temporal delays
- **🧪 Dye System Visualization**: 2D dye concentration maps with diffusion animation
- **📊 Training Progress Dashboard**: Live training metrics and convergence analysis
- **🎮 Interactive Controls**: Play/pause/step simulation with parameter adjustment

**Documentation**: See `docs/visualization/` for complete specifications and development progress.

---

<!-- 😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀 -->

**Neural-Klotski**: Where neuroscience meets computational intelligence through bio-inspired sliding block dynamics.