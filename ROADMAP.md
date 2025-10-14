# Neural-Klotski Development Roadmap

This document tracks the ultra-granular implementation progress of the Neural-Klotski simulation system. Each item represents an atomic component that can be implemented, tested, and validated independently.

## Phase 0: Foundation & Infrastructure ✅ COMPLETED

### ✅ Project Structure (Session 001)
- [x] Create `/neural-klotski/` directory structure
- [x] Copy specification PDF to `docs/specification/`
- [x] Initialize Git repository with proper .gitignore
- [x] Set up Python package structure with `__init__.py`
- [x] Create `requirements.txt` with scientific dependencies

### ✅ Configuration System (Session 001)
- [x] Create `config.py` with all parameters from Section 9.7
- [x] Implement `DynamicsConfig` with validation (dt, mass, damping, spring, refractory)
- [x] Implement `WireConfig` with validation (strength bounds, signal speed, fatigue)
- [x] Implement `DyeConfig` with validation (diffusion, decay, enhancement)
- [x] Implement `LearningConfig` with validation (Hebbian, STDP, threshold adaptation)
- [x] Implement `ThresholdConfig` with validation (ranges for different shelf types)
- [x] Implement `NetworkConfig` with validation (79-block architecture, connectivity)
- [x] Implement `SimulationConfig` with validation (training parameters)
- [x] Add parameter bounds checking and serialization/deserialization

### ✅ Mathematical Utilities (Session 001)
- [x] Create `Vector2D` class with full 2D vector operations
- [x] Implement lag distance calculations
- [x] Implement signal propagation delay calculations
- [x] Add bounds checking and clamping functions
- [x] Implement Euler integration (basic and damped versions)
- [x] Add spring force calculation
- [x] Implement exponential decay function
- [x] Add discrete 2D Laplacian for diffusion
- [x] Implement diffusion step calculation
- [x] Add threshold crossing detection (from below requirement)
- [x] Implement k-nearest neighbors calculation
- [x] Add wire effective strength calculation with dye enhancement

## Phase 1A: Block Foundation ✅ COMPLETED

### ✅ Block State Variables (Session 002)
- [x] Create `BlockState` class with position/velocity on activation axis
- [x] Add threshold and refractory timer properties
- [x] Implement color identity enumeration (Red, Blue, Yellow)
- [x] Add lag position property (fixed during operation)
- [x] Add comprehensive state validation methods

### ✅ Block Physics Engine (Session 002)
- [x] Implement spring force calculation: `F_spring = -k × x`
- [x] Add damping force computation
- [x] Create force accumulation system for multiple wire inputs
- [x] Implement Euler integration for position/velocity updates
- [x] Add position bounds checking (soft bounds)

### ✅ Threshold Detection (Session 002)
- [x] Implement "crossing from below" detection algorithm
- [x] Add refractory state gating logic (no firing when refractory > 0)
- [x] Create firing event generation with proper state transitions
- [x] Implement refractory kick velocity application (leftward impulse)
- [x] Add refractory timer countdown mechanism

## Phase 1B: Wire Foundation 📋 PLANNED

### 📋 Wire Data Structure
- [ ] Create `Wire` class with source/target block references
- [ ] Add strength parameter with bounds enforcement
- [ ] Implement color inheritance from source block
- [ ] Add damage/fatigue state variables
- [ ] Add spatial position property for dye lookup

### 📋 Signal Representation
- [ ] Create `Signal` class with arrival time, strength, color
- [ ] Implement signal queue data structure with temporal ordering
- [ ] Add signal creation from firing events
- [ ] Create signal delivery mechanism to target blocks

### 📋 Propagation Delay Calculation
- [ ] Implement lag distance calculation: `|λ_target - λ_source|`
- [ ] Add propagation speed parameter handling
- [ ] Create delay computation: `delay = distance / speed`
- [ ] Implement signal scheduling system with priority queue

## Phase 1C: Force Application 📋 PLANNED

### 📋 Wire Force Calculation
- [ ] Red wire force: `F = +w_eff` (rightward push)
- [ ] Blue wire force: `F = -w_eff` (leftward push)
- [ ] Yellow wire force: `F = w_eff × (x_source - x_target)` (continuous coupling)
- [ ] Force summation from multiple active wires

### 📋 Signal Queue Management
- [ ] Signal insertion with arrival time sorting
- [ ] Current timestep signal extraction
- [ ] Signal delivery to target blocks with force application
- [ ] Queue cleanup (remove delivered signals)

## Phase 2A: Dye Foundation 📋 PLANNED

### 📋 Dye Field Data Structure
- [ ] Create 2D spatial grid for concentration values
- [ ] Implement separate fields for red/blue/yellow dye
- [ ] Add spatial coordinate mapping (activation × lag space)
- [ ] Create concentration bounds checking (non-negative, max limits)

### 📋 Diffusion Implementation
- [ ] Implement spatial Laplacian: `∇²C` using discrete approximation
- [ ] Add diffusion equation: `∂C/∂t = D × ∇²C`
- [ ] Create neighbor averaging with proper boundary conditions
- [ ] Implement efficient grid update algorithm

### 📋 Decay Implementation
- [ ] Add exponential decay: `C(t) = C(0) × e^(-t/τ)`
- [ ] Implement timestep-based decay updates
- [ ] Create decay rate parameter management
- [ ] Add concentration floor handling (minimum non-zero values)

## Phase 2B: Plasticity Foundation 📋 PLANNED

### 📋 Dye Enhancement Calculation
- [ ] Implement local dye concentration lookup for wires
- [ ] Calculate effective strength: `w_eff = w × (1 + α × C_local)`
- [ ] Add color-selective enhancement (wire color must match dye color)
- [ ] Create enhancement factor bounds checking

### 📋 Hebbian Learning
- [ ] Implement activity correlation detection within temporal window
- [ ] Calculate learning update: `Δw = η × dye_factor × correlation`
- [ ] Add weight bounds enforcement (min/max strength limits)
- [ ] Create learning rate scheduling and adaptation

### 📋 Threshold Adaptation
- [ ] Implement firing rate measurement over sliding window
- [ ] Calculate threshold adjustment based on target firing rate
- [ ] Add homeostatic update: `Δτ = η_τ × (rate_actual - rate_target)`
- [ ] Enforce threshold bounds (min/max values)

## Phase 3A: Network Foundation 📋 PLANNED

### 📋 Connectivity Algorithm
- [ ] Implement K-nearest neighbors in lag space (K=20)
- [ ] Add random long-range connections (L=5, distance >50)
- [ ] Create strength randomization within specified bounds
- [ ] Implement shelf-aware connection patterns (avoid certain cross-shelf connections)

### 📋 Network Initialization
- [ ] Create 4-shelf structure with exact lag positions (Section 8.2)
- [ ] Assign block colors per shelf specifications (input=all red, hidden=mixed, output=all red)
- [ ] Initialize thresholds within specified ranges for each shelf type
- [ ] Apply connectivity algorithm to create wire network
- [ ] Add output layer lateral inhibition (winner-take-all setup)

## Phase 3B: Addition Task 📋 PLANNED

### 📋 Input Encoding
- [ ] Implement one-hot digit encoding (digit d → block d+1 active)
- [ ] Map digits to specific blocks (first 10 blocks for digit 1, next 10 for digit 2)
- [ ] Create input scaling: `position = digit_value × scale_factor`
- [ ] Handle zero encoding correctly (digit 0 → block 1 or block 11)

### 📋 Output Decoding
- [ ] Implement winner-take-all across output blocks (blocks 61-79)
- [ ] Map winning block index to sum value (block 61 → sum 0, block 62 → sum 1, etc.)
- [ ] Add confidence/margin calculations for output reliability
- [ ] Create output validation and error detection

### 📋 Training Protocol
- [ ] Implement trial execution loop (500 timesteps per trial)
- [ ] Add success/failure evaluation (predicted sum vs. actual sum)
- [ ] Create dye injection based on trial outcome (red dye for success)
- [ ] Implement consolidation phase with repeated presentation
- [ ] Add training convergence detection (>95% accuracy criterion)

## Phase 4: Validation & Analysis 📋 PLANNED

### 📋 Mathematical Validation
- [ ] Verify dynamics equations match Section 9.2 exactly
- [ ] Test parameter values against Section 9.7 ranges
- [ ] Validate plasticity rules against Section 9.4 specifications
- [ ] Check architectural specs against Section 8.2 requirements

### 📋 Testing Framework
- [ ] Unit tests for each mathematical component (math_utils)
- [ ] Integration tests for signal propagation timing
- [ ] Network behavior validation tests (firing patterns, connectivity)
- [ ] Generalization experiment implementation (Section 8.9)

## Implementation Progress Summary

### ✅ Completed (Session 001)
- Complete project foundation and infrastructure
- Configuration system with all specification parameters
- Mathematical utilities with full validation
- Session tracking and documentation framework

### 🎯 Current Focus
Phase 1A Complete! Ready to start Phase 1B: Wire Foundation - implementing Wire class and signal propagation

### 📊 Overall Progress: 40% Complete
- Phase 0: 100% ✅
- Phase 1A: 100% ✅
- Phase 1B: 0% 📋
- Phase 1C: 0% 📋
- Phase 2A: 0% 📋
- Phase 2B: 0% 📋
- Phase 3A: 0% 📋
- Phase 3B: 0% 📋
- Phase 4: 0% 📋

## Session Notes

### Session 002 (Current) - Block Foundation
**Date**: October 14, 2024
**Completed**: Phase 1A (Block Foundation)
**Key Achievements**:
- Complete BlockState class with physics integration
- 23 comprehensive unit tests (all passing)
- Threshold crossing and refractory mechanics
- Ready for Phase 1B: Wire Foundation

### Session 001 - Foundation Setup
**Date**: October 14, 2024
**Completed**: Phase 0 (Foundation & Infrastructure)
**Key Achievements**:
- Established complete project structure with proper Git tracking
- Implemented comprehensive configuration system with all specification parameters
- Created mathematical utilities with precision implementations
- Set up session tracking and roadmap system for multi-session development

**Next Session Priority**: Begin Phase 1A with `BlockState` class implementation

---

*This roadmap is designed for multi-session development with Claude Code. Each session should update progress and maintain detailed implementation notes.*