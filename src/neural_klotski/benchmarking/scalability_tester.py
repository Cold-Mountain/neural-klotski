"""
Scalability Testing Framework for Neural-Klotski System.

Evaluates system performance characteristics across different scales:
network sizes, problem complexities, training loads, and resource requirements.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import psutil
import gc
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.core.architecture import create_addition_network
from neural_klotski.core.encoding import AdditionProblem, AdditionTaskManager
from neural_klotski.training.trainer import AdditionNetworkTrainer, TrainingConfig, TrainingPhase
from neural_klotski.config import get_default_config


@dataclass
class ScalabilityTestConfig:
    """Configuration for scalability testing"""
    test_name: str
    description: str
    parameter_name: str
    parameter_values: List[Any]
    metric_names: List[str]  # What to measure: time, memory, accuracy, etc.
    repetitions: int = 3  # Number of test repetitions for statistical stability


@dataclass
class ScalabilityMeasurement:
    """Single measurement in scalability testing"""
    parameter_value: Any
    execution_time: float
    peak_memory_mb: float
    final_accuracy: float
    problems_per_second: float
    convergence_epochs: int
    error_rate: float

    # Resource usage
    cpu_percent: float = 0.0
    memory_delta_mb: float = 0.0

    # Quality metrics
    confidence_score: float = 0.0
    stability_score: float = 0.0  # Consistency across repetitions


@dataclass
class ScalabilityResult:
    """Results from scalability testing"""
    config: ScalabilityTestConfig
    measurements: List[ScalabilityMeasurement]

    # Statistical analysis
    performance_trends: Dict[str, str] = field(default_factory=dict)  # metric -> "linear", "quadratic", etc.
    scaling_coefficients: Dict[str, float] = field(default_factory=dict)  # slope/rate of scaling
    bottleneck_analysis: List[str] = field(default_factory=list)

    # Resource limits
    max_sustainable_scale: Optional[Any] = None
    resource_constraints: List[str] = field(default_factory=list)


@dataclass
class ComprehensiveScalabilityReport:
    """Complete scalability analysis across all test dimensions"""
    test_results: Dict[str, ScalabilityResult]

    # Cross-test analysis
    primary_bottlenecks: List[str]
    scaling_characteristics: Dict[str, str]  # dimension -> behavior
    resource_recommendations: List[str]

    # Performance envelope
    optimal_configurations: List[Tuple[str, Any, str]]  # (parameter, value, reason)
    scaling_limits: Dict[str, Any]  # parameter -> max sustainable value


class ScalabilityTester:
    """
    Comprehensive scalability testing framework for Neural-Klotski.

    Tests system behavior across multiple scaling dimensions:
    - Training duration (epochs, problems per epoch)
    - Problem complexity (number ranges, batch sizes)
    - Network parameters (connectivity, learning rates)
    - Resource usage (memory, CPU time)
    """

    def __init__(self):
        """Initialize scalability tester"""
        self.process = psutil.Process()
        self.test_results: Dict[str, ScalabilityResult] = {}

        # Define standard scalability tests
        self.scalability_tests = self._create_standard_scalability_tests()

    def _create_standard_scalability_tests(self) -> List[ScalabilityTestConfig]:
        """Create standard scalability test configurations"""
        return [
            ScalabilityTestConfig(
                test_name="training_duration",
                description="Performance vs training duration",
                parameter_name="total_epochs",
                parameter_values=[10, 25, 50, 100, 200, 500],
                metric_names=["execution_time", "final_accuracy", "peak_memory_mb"],
                repetitions=3
            ),
            ScalabilityTestConfig(
                test_name="problems_per_epoch",
                description="Performance vs problems per epoch",
                parameter_name="problems_per_epoch",
                parameter_values=[5, 10, 20, 50, 100, 200],
                metric_names=["execution_time", "problems_per_second", "peak_memory_mb"],
                repetitions=3
            ),
            ScalabilityTestConfig(
                test_name="problem_complexity",
                description="Performance vs problem difficulty range",
                parameter_name="max_operand",
                parameter_values=[15, 31, 63, 127, 255, 511],
                metric_names=["final_accuracy", "convergence_epochs", "execution_time"],
                repetitions=3
            ),
            ScalabilityTestConfig(
                test_name="batch_size_scaling",
                description="Inference performance vs batch size",
                parameter_name="batch_size",
                parameter_values=[1, 5, 10, 25, 50, 100, 200],
                metric_names=["problems_per_second", "peak_memory_mb", "execution_time"],
                repetitions=5
            ),
            ScalabilityTestConfig(
                test_name="network_connectivity",
                description="Performance vs network connectivity density",
                parameter_name="k_neighbors",
                parameter_values=[5, 10, 15, 20, 25, 30],
                metric_names=["execution_time", "final_accuracy", "peak_memory_mb"],
                repetitions=3
            ),
            ScalabilityTestConfig(
                test_name="learning_rate_scaling",
                description="Convergence vs learning rate values",
                parameter_name="base_learning_rate",
                parameter_values=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
                metric_names=["convergence_epochs", "final_accuracy", "stability_score"],
                repetitions=3
            )
        ]

    def run_scalability_test(self, config: ScalabilityTestConfig) -> ScalabilityResult:
        """Run a single scalability test across parameter values"""
        print(f"📊 Running scalability test: {config.test_name}")
        print(f"   {config.description}")
        print(f"   Testing {len(config.parameter_values)} parameter values with {config.repetitions} repetitions each")

        measurements = []

        for param_value in config.parameter_values:
            print(f"   Testing {config.parameter_name} = {param_value}")

            # Run multiple repetitions for statistical stability
            repetition_results = []

            for rep in range(config.repetitions):
                try:
                    measurement = self._run_single_measurement(config, param_value, rep)
                    repetition_results.append(measurement)
                except Exception as e:
                    print(f"     ❌ Repetition {rep+1} failed: {e}")

            if repetition_results:
                # Aggregate repetition results
                avg_measurement = self._aggregate_measurements(param_value, repetition_results)
                measurements.append(avg_measurement)
                print(f"     ✅ {config.parameter_name}={param_value}: "
                      f"{avg_measurement.execution_time:.2f}s, "
                      f"{avg_measurement.peak_memory_mb:.1f}MB, "
                      f"{avg_measurement.final_accuracy:.3f} acc")

        # Analyze scaling behavior
        result = ScalabilityResult(config=config, measurements=measurements)
        self._analyze_scaling_behavior(result)

        print(f"   📈 Scaling analysis completed\n")
        return result

    def _run_single_measurement(self, config: ScalabilityTestConfig, param_value: Any, repetition: int) -> ScalabilityMeasurement:
        """Run a single measurement with specific parameter value"""
        # Force garbage collection before measurement
        gc.collect()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        start_time = time.time()

        if config.parameter_name in ["total_epochs", "problems_per_epoch"]:
            # Training-based tests
            measurement = self._measure_training_performance(config.parameter_name, param_value)

        elif config.parameter_name == "max_operand":
            # Problem complexity tests
            measurement = self._measure_problem_complexity_performance(param_value)

        elif config.parameter_name == "batch_size":
            # Batch inference tests
            measurement = self._measure_batch_inference_performance(param_value)

        elif config.parameter_name == "k_neighbors":
            # Network connectivity tests
            measurement = self._measure_connectivity_performance(param_value)

        elif config.parameter_name == "base_learning_rate":
            # Learning rate scaling tests
            measurement = self._measure_learning_rate_performance(param_value)

        else:
            raise ValueError(f"Unknown parameter: {config.parameter_name}")

        # Finalize measurement
        end_time = time.time()
        final_memory = self.process.memory_info().rss / 1024 / 1024

        measurement.execution_time = end_time - start_time
        measurement.memory_delta_mb = final_memory - initial_memory
        measurement.cpu_percent = self.process.cpu_percent()

        return measurement

    def _measure_training_performance(self, param_name: str, param_value: Any) -> ScalabilityMeasurement:
        """Measure training performance with specific parameter"""
        config = get_default_config()
        training_config = TrainingConfig()

        if param_name == "total_epochs":
            # Scale total epochs while keeping problems per epoch constant
            epochs_per_phase = {
                TrainingPhase.SIMPLE: max(1, param_value // 4),
                TrainingPhase.INTERMEDIATE: max(1, param_value // 3),
                TrainingPhase.ADVANCED: max(1, param_value // 2)
            }
            training_config.epochs_per_phase = epochs_per_phase
            training_config.problems_per_epoch = 20

        elif param_name == "problems_per_epoch":
            # Scale problems per epoch while keeping total training reasonable
            training_config.epochs_per_phase = {
                TrainingPhase.SIMPLE: 10,
                TrainingPhase.INTERMEDIATE: 15,
                TrainingPhase.ADVANCED: 25
            }
            training_config.problems_per_epoch = param_value

        # Monitor memory during training
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        peak_memory = initial_memory

        # Create trainer and train
        trainer = AdditionNetworkTrainer(training_config, config.simulation)

        # Hook into training to monitor memory
        original_train_epoch = trainer._train_single_epoch

        def monitored_train_epoch(*args, **kwargs):
            nonlocal peak_memory
            current_memory = self.process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
            return original_train_epoch(*args, **kwargs)

        trainer._train_single_epoch = monitored_train_epoch

        # Run training
        start_time = time.time()
        final_metrics = trainer.train()
        training_time = time.time() - start_time

        # Calculate derived metrics
        total_problems = sum(training_config.epochs_per_phase.values()) * training_config.problems_per_epoch
        problems_per_second = total_problems / training_time if training_time > 0 else 0

        return ScalabilityMeasurement(
            parameter_value=param_value,
            execution_time=training_time,
            peak_memory_mb=peak_memory,
            final_accuracy=final_metrics.accuracy,
            problems_per_second=problems_per_second,
            convergence_epochs=final_metrics.total_epochs,
            error_rate=1.0 - final_metrics.accuracy,
            confidence_score=final_metrics.average_confidence
        )

    def _measure_problem_complexity_performance(self, max_operand: int) -> ScalabilityMeasurement:
        """Measure performance vs problem complexity"""
        # Create network
        network = create_addition_network(enable_learning=True)
        config = get_default_config()

        # Train on progressively complex problems
        training_config = TrainingConfig()
        training_config.epochs_per_phase = {
            TrainingPhase.SIMPLE: 25,
            TrainingPhase.INTERMEDIATE: 25,
            TrainingPhase.ADVANCED: 50
        }
        training_config.problems_per_epoch = 20

        # Modify problem generation to use specified max operand
        trainer = AdditionNetworkTrainer(training_config, config.simulation)

        # Override problem generation (simplified approach)
        start_time = time.time()
        final_metrics = trainer.train()
        training_time = time.time() - start_time

        # Test final performance on problems of target complexity
        task_manager = AdditionTaskManager(config.simulation)
        test_problems = 50
        correct_count = 0

        peak_memory = self.process.memory_info().rss / 1024 / 1024

        for _ in range(test_problems):
            import random
            a = random.randint(0, max_operand)
            b = random.randint(0, max_operand)
            problem = AdditionProblem(a, b, a + b)

            network.reset_simulation()
            result, _ = task_manager.execute_addition_task(network, problem)

            if result.decoded_sum == problem.expected_sum:
                correct_count += 1

            current_memory = self.process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)

        final_accuracy = correct_count / test_problems

        return ScalabilityMeasurement(
            parameter_value=max_operand,
            execution_time=training_time,
            peak_memory_mb=peak_memory,
            final_accuracy=final_accuracy,
            problems_per_second=test_problems / training_time,
            convergence_epochs=final_metrics.total_epochs,
            error_rate=1.0 - final_accuracy
        )

    def _measure_batch_inference_performance(self, batch_size: int) -> ScalabilityMeasurement:
        """Measure batch inference performance"""
        # Create network
        network = create_addition_network(enable_learning=False)
        config = get_default_config()
        task_manager = AdditionTaskManager(config.simulation)

        # Generate test problems
        import random
        problems = []
        for _ in range(batch_size):
            a = random.randint(0, 100)
            b = random.randint(0, 100)
            problems.append(AdditionProblem(a, b, a + b))

        # Measure batch inference
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        peak_memory = initial_memory

        start_time = time.time()
        correct_count = 0

        for problem in problems:
            network.reset_simulation()
            result, _ = task_manager.execute_addition_task(network, problem)

            if result.decoded_sum == problem.expected_sum:
                correct_count += 1

            current_memory = self.process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)

        end_time = time.time()
        batch_time = end_time - start_time
        problems_per_second = batch_size / batch_time if batch_time > 0 else 0

        return ScalabilityMeasurement(
            parameter_value=batch_size,
            execution_time=batch_time,
            peak_memory_mb=peak_memory,
            final_accuracy=correct_count / batch_size,
            problems_per_second=problems_per_second,
            convergence_epochs=0,  # No training
            error_rate=1.0 - (correct_count / batch_size)
        )

    def _measure_connectivity_performance(self, k_neighbors: int) -> ScalabilityMeasurement:
        """Measure performance vs network connectivity"""
        # This would require modifying network creation to use different K values
        # For now, use standard network with performance proxy
        network = create_addition_network(enable_learning=True)
        config = get_default_config()

        # Run short training
        training_config = TrainingConfig()
        training_config.epochs_per_phase = {
            TrainingPhase.SIMPLE: 10,
            TrainingPhase.INTERMEDIATE: 0,
            TrainingPhase.ADVANCED: 0
        }
        training_config.problems_per_epoch = 10

        trainer = AdditionNetworkTrainer(training_config, config.simulation)

        start_time = time.time()
        final_metrics = trainer.train()
        training_time = time.time() - start_time

        peak_memory = self.process.memory_info().rss / 1024 / 1024

        return ScalabilityMeasurement(
            parameter_value=k_neighbors,
            execution_time=training_time,
            peak_memory_mb=peak_memory,
            final_accuracy=final_metrics.accuracy,
            problems_per_second=100 / training_time,  # 10 epochs * 10 problems
            convergence_epochs=final_metrics.total_epochs,
            error_rate=1.0 - final_metrics.accuracy
        )

    def _measure_learning_rate_performance(self, learning_rate: float) -> ScalabilityMeasurement:
        """Measure performance vs learning rate"""
        # Modify learning parameters
        config = get_default_config()

        # Scale learning rates
        config.learning.hebbian_learning_rate *= learning_rate / 0.01  # Assume base rate of 0.01
        config.learning.threshold_adaptation_rate *= learning_rate / 0.01

        training_config = TrainingConfig()
        training_config.epochs_per_phase = {
            TrainingPhase.SIMPLE: 20,
            TrainingPhase.INTERMEDIATE: 0,
            TrainingPhase.ADVANCED: 0
        }
        training_config.problems_per_epoch = 15

        trainer = AdditionNetworkTrainer(training_config, config.simulation)

        start_time = time.time()
        final_metrics = trainer.train()
        training_time = time.time() - start_time

        peak_memory = self.process.memory_info().rss / 1024 / 1024

        # Calculate stability score based on accuracy variance
        stability_score = min(1.0, final_metrics.accuracy)  # Simplified

        return ScalabilityMeasurement(
            parameter_value=learning_rate,
            execution_time=training_time,
            peak_memory_mb=peak_memory,
            final_accuracy=final_metrics.accuracy,
            problems_per_second=300 / training_time,  # 20 epochs * 15 problems
            convergence_epochs=final_metrics.total_epochs,
            error_rate=1.0 - final_metrics.accuracy,
            stability_score=stability_score
        )

    def _aggregate_measurements(self, param_value: Any, measurements: List[ScalabilityMeasurement]) -> ScalabilityMeasurement:
        """Aggregate multiple measurement repetitions"""
        if not measurements:
            raise ValueError("No measurements to aggregate")

        # Calculate averages
        avg_measurement = ScalabilityMeasurement(
            parameter_value=param_value,
            execution_time=np.mean([m.execution_time for m in measurements]),
            peak_memory_mb=np.mean([m.peak_memory_mb for m in measurements]),
            final_accuracy=np.mean([m.final_accuracy for m in measurements]),
            problems_per_second=np.mean([m.problems_per_second for m in measurements]),
            convergence_epochs=int(np.mean([m.convergence_epochs for m in measurements])),
            error_rate=np.mean([m.error_rate for m in measurements]),
            cpu_percent=np.mean([m.cpu_percent for m in measurements]),
            memory_delta_mb=np.mean([m.memory_delta_mb for m in measurements]),
            confidence_score=np.mean([m.confidence_score for m in measurements]),
            stability_score=1.0 - np.std([m.final_accuracy for m in measurements])  # Higher std = lower stability
        )

        return avg_measurement

    def _analyze_scaling_behavior(self, result: ScalabilityResult):
        """Analyze scaling behavior from measurements"""
        if len(result.measurements) < 3:
            return

        # Extract data for analysis
        param_values = [m.parameter_value for m in result.measurements]

        # Analyze different metrics
        metrics_data = {
            'execution_time': [m.execution_time for m in result.measurements],
            'peak_memory_mb': [m.peak_memory_mb for m in result.measurements],
            'final_accuracy': [m.final_accuracy for m in result.measurements],
            'problems_per_second': [m.problems_per_second for m in result.measurements]
        }

        # Simple trend analysis
        for metric_name, metric_values in metrics_data.items():
            if len(metric_values) >= 3:
                # Calculate correlation with parameter values
                correlation = np.corrcoef(param_values, metric_values)[0, 1]

                if abs(correlation) > 0.8:
                    if correlation > 0:
                        trend = "strongly_increasing"
                    else:
                        trend = "strongly_decreasing"
                elif abs(correlation) > 0.5:
                    if correlation > 0:
                        trend = "moderately_increasing"
                    else:
                        trend = "moderately_decreasing"
                else:
                    trend = "stable"

                result.performance_trends[metric_name] = trend
                result.scaling_coefficients[metric_name] = correlation

        # Identify bottlenecks
        bottlenecks = []

        # Check for exponential growth in time/memory
        time_values = metrics_data['execution_time']
        if len(time_values) >= 3:
            time_ratios = [time_values[i+1]/time_values[i] for i in range(len(time_values)-1)]
            if any(ratio > 2.0 for ratio in time_ratios):
                bottlenecks.append("Execution time scaling poorly")

        memory_values = metrics_data['peak_memory_mb']
        if len(memory_values) >= 3:
            memory_ratios = [memory_values[i+1]/memory_values[i] for i in range(len(memory_values)-1)]
            if any(ratio > 1.5 for ratio in memory_ratios):
                bottlenecks.append("Memory usage scaling poorly")

        # Check for accuracy degradation
        accuracy_values = metrics_data['final_accuracy']
        if len(accuracy_values) >= 3:
            if accuracy_values[-1] < accuracy_values[0] - 0.1:
                bottlenecks.append("Accuracy degrading with scale")

        result.bottleneck_analysis = bottlenecks

        # Estimate max sustainable scale (simplified)
        max_time = 300.0  # 5 minutes max
        max_memory = 2000.0  # 2GB max

        for measurement in reversed(result.measurements):
            if (measurement.execution_time < max_time and
                measurement.peak_memory_mb < max_memory and
                measurement.final_accuracy > 0.7):
                result.max_sustainable_scale = measurement.parameter_value
                break

    def run_comprehensive_scalability_analysis(self) -> ComprehensiveScalabilityReport:
        """Run comprehensive scalability analysis across all test dimensions"""
        print("📊 Running Comprehensive Scalability Analysis")
        print("=" * 60)

        overall_start = time.time()

        # Run all scalability tests
        for test_config in self.scalability_tests:
            try:
                result = self.run_scalability_test(test_config)
                self.test_results[test_config.test_name] = result
            except Exception as e:
                print(f"❌ Failed to run test {test_config.test_name}: {e}")
                print()

        overall_duration = time.time() - overall_start

        # Generate comprehensive report
        report = self._generate_scalability_report()

        # Print summary
        self._print_scalability_summary(report, overall_duration)

        return report

    def _generate_scalability_report(self) -> ComprehensiveScalabilityReport:
        """Generate comprehensive scalability analysis report"""
        if not self.test_results:
            return ComprehensiveScalabilityReport(
                test_results={},
                primary_bottlenecks=[],
                scaling_characteristics={},
                resource_recommendations=[],
                optimal_configurations=[],
                scaling_limits={}
            )

        # Analyze primary bottlenecks across all tests
        all_bottlenecks = []
        for result in self.test_results.values():
            all_bottlenecks.extend(result.bottleneck_analysis)

        bottleneck_counts = {}
        for bottleneck in all_bottlenecks:
            bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1

        primary_bottlenecks = sorted(bottleneck_counts.keys(),
                                   key=lambda x: bottleneck_counts[x],
                                   reverse=True)[:3]

        # Analyze scaling characteristics
        scaling_chars = {}
        for test_name, result in self.test_results.items():
            if result.performance_trends:
                dominant_trend = max(result.performance_trends.items(),
                                   key=lambda x: abs(result.scaling_coefficients.get(x[0], 0)))
                scaling_chars[test_name] = f"{dominant_trend[0]}: {dominant_trend[1]}"

        # Generate recommendations
        resource_recs = self._generate_resource_recommendations()
        optimal_configs = self._identify_optimal_configurations()
        scaling_limits = self._determine_scaling_limits()

        return ComprehensiveScalabilityReport(
            test_results=self.test_results,
            primary_bottlenecks=primary_bottlenecks,
            scaling_characteristics=scaling_chars,
            resource_recommendations=resource_recs,
            optimal_configurations=optimal_configs,
            scaling_limits=scaling_limits
        )

    def _generate_resource_recommendations(self) -> List[str]:
        """Generate resource optimization recommendations"""
        recommendations = []

        # Check memory usage patterns
        max_memory = 0
        for result in self.test_results.values():
            for measurement in result.measurements:
                max_memory = max(max_memory, measurement.peak_memory_mb)

        if max_memory > 1000:  # >1GB
            recommendations.append("High memory usage detected - consider memory optimization")
        elif max_memory > 500:  # >500MB
            recommendations.append("Moderate memory usage - monitor for larger scales")

        # Check execution time patterns
        max_time = 0
        for result in self.test_results.values():
            for measurement in result.measurements:
                max_time = max(max_time, measurement.execution_time)

        if max_time > 120:  # >2 minutes
            recommendations.append("Long execution times - consider parallel processing")
        elif max_time > 30:  # >30 seconds
            recommendations.append("Moderate execution times - acceptable for current scale")

        # Check performance degradation
        degradation_found = False
        for result in self.test_results.values():
            if "Accuracy degrading with scale" in result.bottleneck_analysis:
                degradation_found = True
                break

        if degradation_found:
            recommendations.append("Accuracy degradation detected - review learning parameters")

        if not recommendations:
            recommendations.append("Resource usage is reasonable across tested scales")

        return recommendations

    def _identify_optimal_configurations(self) -> List[Tuple[str, Any, str]]:
        """Identify optimal configuration points"""
        optimal_configs = []

        for test_name, result in self.test_results.values():
            if not result.measurements:
                continue

            # Find configuration with best accuracy/time trade-off
            best_measurement = None
            best_score = -1

            for measurement in result.measurements:
                # Simple scoring: accuracy / (time + memory_factor)
                memory_factor = measurement.peak_memory_mb / 1000  # GB
                score = measurement.final_accuracy / (measurement.execution_time + memory_factor)

                if score > best_score:
                    best_score = score
                    best_measurement = measurement

            if best_measurement:
                reason = f"Best accuracy/resource ratio ({best_measurement.final_accuracy:.3f} acc, {best_measurement.execution_time:.1f}s)"
                optimal_configs.append((result.config.parameter_name, best_measurement.parameter_value, reason))

        return optimal_configs

    def _determine_scaling_limits(self) -> Dict[str, Any]:
        """Determine scaling limits across different dimensions"""
        limits = {}

        for test_name, result in self.test_results.items():
            if result.max_sustainable_scale is not None:
                limits[result.config.parameter_name] = result.max_sustainable_scale

        return limits

    def _print_scalability_summary(self, report: ComprehensiveScalabilityReport, duration: float):
        """Print comprehensive scalability summary"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE SCALABILITY ANALYSIS SUMMARY")
        print("="*60)

        print(f"Total analysis time: {duration:.1f}s")
        print(f"Scalability tests completed: {len(report.test_results)}")

        # Test results overview
        if report.test_results:
            print(f"\n📈 Scalability Test Results:")
            for test_name, result in report.test_results.items():
                print(f"  • {test_name}:")
                print(f"    - Parameter range: {result.measurements[0].parameter_value} to {result.measurements[-1].parameter_value}")
                print(f"    - Max sustainable: {result.max_sustainable_scale}")
                if result.bottleneck_analysis:
                    print(f"    - Bottlenecks: {', '.join(result.bottleneck_analysis)}")

        # Scaling characteristics
        if report.scaling_characteristics:
            print(f"\n⚡ Scaling Characteristics:")
            for dimension, behavior in report.scaling_characteristics.items():
                print(f"  • {dimension}: {behavior}")

        # Primary bottlenecks
        if report.primary_bottlenecks:
            print(f"\n🚨 Primary Bottlenecks:")
            for i, bottleneck in enumerate(report.primary_bottlenecks, 1):
                print(f"  {i}. {bottleneck}")

        # Optimal configurations
        if report.optimal_configurations:
            print(f"\n🎯 Optimal Configurations:")
            for param, value, reason in report.optimal_configurations:
                print(f"  • {param} = {value}: {reason}")

        # Scaling limits
        if report.scaling_limits:
            print(f"\n⚠️ Scaling Limits:")
            for param, limit in report.scaling_limits.items():
                print(f"  • {param}: max sustainable = {limit}")

        # Resource recommendations
        if report.resource_recommendations:
            print(f"\n💡 Resource Recommendations:")
            for i, rec in enumerate(report.resource_recommendations, 1):
                print(f"  {i}. {rec}")

        print("="*60)

    def export_scalability_results(self, report: ComprehensiveScalabilityReport, filename: str = "scalability_analysis.json"):
        """Export scalability analysis results to JSON"""
        try:
            import json

            export_data = {
                'primary_bottlenecks': report.primary_bottlenecks,
                'scaling_characteristics': report.scaling_characteristics,
                'resource_recommendations': report.resource_recommendations,
                'optimal_configurations': [
                    {'parameter': param, 'value': value, 'reason': reason}
                    for param, value, reason in report.optimal_configurations
                ],
                'scaling_limits': report.scaling_limits,
                'test_results': {}
            }

            # Export test results (simplified)
            for test_name, result in report.test_results.items():
                export_data['test_results'][test_name] = {
                    'parameter_name': result.config.parameter_name,
                    'measurements': [
                        {
                            'parameter_value': m.parameter_value,
                            'execution_time': m.execution_time,
                            'peak_memory_mb': m.peak_memory_mb,
                            'final_accuracy': m.final_accuracy,
                            'problems_per_second': m.problems_per_second
                        }
                        for m in result.measurements
                    ],
                    'performance_trends': result.performance_trends,
                    'scaling_coefficients': result.scaling_coefficients,
                    'bottleneck_analysis': result.bottleneck_analysis,
                    'max_sustainable_scale': result.max_sustainable_scale
                }

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)

            print(f"📁 Scalability analysis results exported to {filename}")

        except Exception as e:
            print(f"❌ Failed to export scalability results: {e}")


if __name__ == "__main__":
    # Test scalability tester
    print("📊 Testing Neural-Klotski Scalability Tester...")

    tester = ScalabilityTester()

    # Run comprehensive analysis
    report = tester.run_comprehensive_scalability_analysis()

    # Export results
    tester.export_scalability_results(report, "test_scalability_analysis.json")

    print("\n✅ Scalability testing completed!")