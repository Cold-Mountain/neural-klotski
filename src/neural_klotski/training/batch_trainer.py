"""
Batch Training System for Neural-Klotski Networks.

Supports multiple training runs, hyperparameter sweeps, and
comprehensive experiment management.
"""

from typing import List, Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time
import concurrent.futures
import itertools
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.training.trainer import (
    AdditionNetworkTrainer, TrainingConfig, TrainingMetrics, TrainingPhase
)
from neural_klotski.config import SimulationConfig, get_default_config


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    description: str = ""

    # Training parameters to vary
    training_params: Dict[str, Any] = field(default_factory=dict)
    simulation_params: Dict[str, Any] = field(default_factory=dict)

    # Experiment metadata
    seed: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    expected_duration_minutes: float = 30.0


@dataclass
class ExperimentResult:
    """Result of a single experiment"""
    config: ExperimentConfig
    final_metrics: TrainingMetrics
    success: bool
    error_message: Optional[str] = None

    # Performance summary
    peak_accuracy: float = 0.0
    convergence_epoch: Optional[int] = None
    total_duration: float = 0.0

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    """Configuration for batch training"""
    output_directory: str = "experiments"
    max_parallel_jobs: int = 1
    save_detailed_logs: bool = True
    continue_on_failure: bool = True

    # Resource limits
    max_experiment_duration_hours: float = 2.0
    memory_limit_gb: Optional[float] = None


class HyperparameterSweep:
    """
    Generates hyperparameter combinations for systematic exploration.

    Supports grid search, random sampling, and Bayesian optimization
    preparation.
    """

    def __init__(self):
        self.parameter_space = {}

    def add_parameter(self, name: str, values: List[Any]):
        """Add parameter with discrete values"""
        self.parameter_space[name] = values

    def add_range(self, name: str, min_val: float, max_val: float,
                  num_points: int, log_scale: bool = False):
        """Add parameter with continuous range"""
        import numpy as np

        if log_scale:
            values = np.logspace(np.log10(min_val), np.log10(max_val), num_points)
        else:
            values = np.linspace(min_val, max_val, num_points)

        self.parameter_space[name] = values.tolist()

    def grid_search(self) -> Iterator[Dict[str, Any]]:
        """Generate all combinations (grid search)"""
        param_names = list(self.parameter_space.keys())
        param_values = [self.parameter_space[name] for name in param_names]

        for combination in itertools.product(*param_values):
            yield dict(zip(param_names, combination))

    def random_sample(self, num_samples: int, seed: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Generate random samples from parameter space"""
        import random

        if seed is not None:
            random.seed(seed)

        param_names = list(self.parameter_space.keys())

        for _ in range(num_samples):
            combination = {}
            for name in param_names:
                combination[name] = random.choice(self.parameter_space[name])
            yield combination


class BatchTrainer:
    """
    Manages batch training of multiple Neural-Klotski networks.

    Supports hyperparameter sweeps, parallel execution, and
    comprehensive result tracking.
    """

    def __init__(self, config: BatchConfig):
        """
        Initialize batch trainer.

        Args:
            config: Batch training configuration
        """
        self.config = config
        self.results: List[ExperimentResult] = []

        # Create output directory
        self.output_path = Path(config.output_directory)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Experiment tracking
        self.start_time = time.time()
        self.experiments_completed = 0
        self.experiments_failed = 0

    def create_experiment(self, name: str, description: str = "",
                         training_overrides: Optional[Dict[str, Any]] = None,
                         simulation_overrides: Optional[Dict[str, Any]] = None,
                         **kwargs) -> ExperimentConfig:
        """
        Create experiment configuration.

        Args:
            name: Experiment name
            description: Experiment description
            training_overrides: Training parameter overrides
            simulation_overrides: Simulation parameter overrides
            **kwargs: Additional experiment metadata

        Returns:
            Experiment configuration
        """
        return ExperimentConfig(
            name=name,
            description=description,
            training_params=training_overrides or {},
            simulation_params=simulation_overrides or {},
            **kwargs
        )

    def run_experiment(self, experiment: ExperimentConfig) -> ExperimentResult:
        """
        Run single experiment.

        Args:
            experiment: Experiment configuration

        Returns:
            Experiment result
        """
        start_time = time.time()

        try:
            print(f"Starting experiment: {experiment.name}")

            # Create configurations
            base_config = get_default_config()
            training_config = TrainingConfig()

            # Apply parameter overrides
            for param, value in experiment.training_params.items():
                if hasattr(training_config, param):
                    setattr(training_config, param, value)
                else:
                    print(f"Warning: Unknown training parameter: {param}")

            for param, value in experiment.simulation_params.items():
                if hasattr(base_config.simulation, param):
                    setattr(base_config.simulation, param, value)
                else:
                    print(f"Warning: Unknown simulation parameter: {param}")

            # Set random seed if specified
            if experiment.seed is not None:
                import random
                import numpy as np
                random.seed(experiment.seed)
                np.random.seed(experiment.seed)

            # Create trainer and run training
            trainer = AdditionNetworkTrainer(training_config, base_config.simulation)

            # Add logging callback for detailed tracking
            metrics_history = []
            def track_metrics(metrics: TrainingMetrics):
                metrics_history.append(asdict(metrics))

            trainer.add_epoch_callback(track_metrics)

            # Run training with timeout
            final_metrics = trainer.train()

            # Calculate performance summary
            peak_accuracy = max(final_metrics.accuracy_history) if final_metrics.accuracy_history else 0.0
            convergence_epoch = None

            # Find convergence point (where accuracy stops improving significantly)
            if len(final_metrics.accuracy_history) > 10:
                for i in range(10, len(final_metrics.accuracy_history)):
                    recent_improvement = (final_metrics.accuracy_history[i] -
                                        final_metrics.accuracy_history[i-10])
                    if recent_improvement < 0.01:  # Less than 1% improvement in 10 epochs
                        convergence_epoch = i
                        break

            total_duration = time.time() - start_time

            # Create result
            result = ExperimentResult(
                config=experiment,
                final_metrics=final_metrics,
                success=True,
                peak_accuracy=peak_accuracy,
                convergence_epoch=convergence_epoch,
                total_duration=total_duration,
                metadata={
                    'metrics_history': metrics_history,
                    'final_phase': final_metrics.phase.value,
                    'total_problems_attempted': final_metrics.total_problems
                }
            )

            print(f"Completed experiment: {experiment.name} "
                  f"(Accuracy: {final_metrics.accuracy:.3f}, "
                  f"Duration: {total_duration:.1f}s)")

            return result

        except Exception as e:
            error_duration = time.time() - start_time
            print(f"Failed experiment: {experiment.name} - {str(e)}")

            return ExperimentResult(
                config=experiment,
                final_metrics=TrainingMetrics(),  # Empty metrics
                success=False,
                error_message=str(e),
                total_duration=error_duration
            )

    def run_hyperparameter_sweep(self, base_name: str, sweep: HyperparameterSweep,
                                max_experiments: Optional[int] = None,
                                method: str = "grid") -> List[ExperimentResult]:
        """
        Run hyperparameter sweep.

        Args:
            base_name: Base name for experiments
            sweep: Hyperparameter sweep configuration
            max_experiments: Maximum number of experiments
            method: Search method ("grid" or "random")

        Returns:
            List of experiment results
        """
        print(f"Starting hyperparameter sweep: {base_name}")

        # Generate parameter combinations
        if method == "grid":
            param_combinations = list(sweep.grid_search())
        elif method == "random":
            if max_experiments is None:
                raise ValueError("max_experiments required for random search")
            param_combinations = list(sweep.random_sample(max_experiments))
        else:
            raise ValueError(f"Unknown search method: {method}")

        if max_experiments is not None:
            param_combinations = param_combinations[:max_experiments]

        print(f"Generated {len(param_combinations)} parameter combinations")

        # Create experiments
        experiments = []
        for i, params in enumerate(param_combinations):
            # Split parameters into training and simulation categories
            training_params = {}
            simulation_params = {}

            for param, value in params.items():
                if param.startswith('sim_'):
                    simulation_params[param[4:]] = value  # Remove 'sim_' prefix
                else:
                    training_params[param] = value

            experiment = self.create_experiment(
                name=f"{base_name}_sweep_{i:03d}",
                description=f"Hyperparameter sweep experiment {i+1}/{len(param_combinations)}",
                training_overrides=training_params,
                simulation_overrides=simulation_params,
                tags=["hyperparameter_sweep", base_name]
            )
            experiments.append(experiment)

        # Run experiments
        return self.run_batch(experiments)

    def run_batch(self, experiments: List[ExperimentConfig]) -> List[ExperimentResult]:
        """
        Run batch of experiments.

        Args:
            experiments: List of experiment configurations

        Returns:
            List of experiment results
        """
        print(f"Running batch of {len(experiments)} experiments...")

        results = []

        if self.config.max_parallel_jobs == 1:
            # Sequential execution
            for experiment in experiments:
                result = self.run_experiment(experiment)
                results.append(result)
                self._update_progress(result)

                if not self.config.continue_on_failure and not result.success:
                    print("Stopping batch due to failure")
                    break
        else:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_parallel_jobs
            ) as executor:
                future_to_experiment = {
                    executor.submit(self.run_experiment, exp): exp
                    for exp in experiments
                }

                for future in concurrent.futures.as_completed(future_to_experiment):
                    result = future.result()
                    results.append(result)
                    self._update_progress(result)

                    if not self.config.continue_on_failure and not result.success:
                        print("Cancelling remaining experiments due to failure")
                        for remaining_future in future_to_experiment:
                            remaining_future.cancel()
                        break

        # Save results
        self._save_batch_results(results)

        print(f"Batch completed: {len(results)} experiments, "
              f"{sum(1 for r in results if r.success)} successful")

        return results

    def _update_progress(self, result: ExperimentResult):
        """Update progress tracking"""
        if result.success:
            self.experiments_completed += 1
        else:
            self.experiments_failed += 1

        self.results.append(result)

    def _save_batch_results(self, results: List[ExperimentResult]):
        """Save batch results to disk"""
        if not self.config.save_detailed_logs:
            return

        timestamp = int(time.time())
        results_file = self.output_path / f"batch_results_{timestamp}.json"

        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            # Convert enum values to strings
            if 'final_metrics' in result_dict and 'phase' in result_dict['final_metrics']:
                result_dict['final_metrics']['phase'] = result_dict['final_metrics']['phase'].value
            serializable_results.append(result_dict)

        with open(results_file, 'w') as f:
            json.dump({
                'batch_config': asdict(self.config),
                'start_time': self.start_time,
                'total_experiments': len(results),
                'successful_experiments': sum(1 for r in results if r.success),
                'results': serializable_results
            }, f, indent=2)

        print(f"Results saved to: {results_file}")

    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze batch results and generate summary.

        Returns:
            Analysis summary
        """
        if not self.results:
            return {"error": "No results to analyze"}

        successful_results = [r for r in self.results if r.success]

        if not successful_results:
            return {"error": "No successful experiments"}

        # Basic statistics
        accuracies = [r.final_metrics.accuracy for r in successful_results]
        durations = [r.total_duration for r in successful_results]

        analysis = {
            'total_experiments': len(self.results),
            'successful_experiments': len(successful_results),
            'success_rate': len(successful_results) / len(self.results),
            'accuracy_stats': {
                'mean': sum(accuracies) / len(accuracies),
                'max': max(accuracies),
                'min': min(accuracies),
                'std': (sum((x - sum(accuracies)/len(accuracies))**2 for x in accuracies) / len(accuracies))**0.5
            },
            'duration_stats': {
                'mean': sum(durations) / len(durations),
                'max': max(durations),
                'min': min(durations)
            },
            'best_experiment': max(successful_results, key=lambda r: r.final_metrics.accuracy).config.name
        }

        # Parameter correlation analysis (if multiple experiments)
        if len(successful_results) > 1:
            analysis['parameter_analysis'] = self._analyze_parameter_effects(successful_results)

        return analysis

    def _analyze_parameter_effects(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze the effect of different parameters on performance"""
        # Group results by parameter values
        param_effects = {}

        for result in results:
            for param, value in result.config.training_params.items():
                if param not in param_effects:
                    param_effects[param] = {}

                if str(value) not in param_effects[param]:
                    param_effects[param][str(value)] = []

                param_effects[param][str(value)].append(result.final_metrics.accuracy)

        # Calculate mean accuracy for each parameter value
        param_summary = {}
        for param, value_results in param_effects.items():
            if len(value_results) > 1:  # Only analyze parameters that were varied
                param_summary[param] = {}
                for value, accuracies in value_results.items():
                    param_summary[param][value] = {
                        'mean_accuracy': sum(accuracies) / len(accuracies),
                        'count': len(accuracies)
                    }

        return param_summary


if __name__ == "__main__":
    # Test batch training system
    print("Testing Neural-Klotski Batch Training System...")

    # Create batch configuration
    batch_config = BatchConfig(
        output_directory="test_experiments",
        max_parallel_jobs=1,
        save_detailed_logs=True
    )

    batch_trainer = BatchTrainer(batch_config)

    # Create a small hyperparameter sweep
    sweep = HyperparameterSweep()
    sweep.add_parameter('problems_per_epoch', [10, 20])
    sweep.add_parameter('sim_input_scaling_factor', [50.0, 100.0])

    # Run sweep
    results = batch_trainer.run_hyperparameter_sweep(
        base_name="test_sweep",
        sweep=sweep,
        max_experiments=4,
        method="grid"
    )

    # Analyze results
    analysis = batch_trainer.analyze_results()
    print(f"Batch analysis: {analysis}")

    print("Batch training system test completed!")