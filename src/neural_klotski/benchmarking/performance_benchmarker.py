"""
Performance Benchmarking Framework for Neural-Klotski System.

Provides comprehensive benchmarking of training convergence, computational
efficiency, accuracy performance, and scalability characteristics.
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import psutil
import numpy as np
import gc
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.core.architecture import create_addition_network
from neural_klotski.core.encoding import AdditionProblem, AdditionTaskManager
from neural_klotski.training.trainer import AdditionNetworkTrainer, TrainingConfig, TrainingPhase
from neural_klotski.training.monitoring import PerformanceMonitor, ConvergenceConfig
from neural_klotski.config import get_default_config


class BenchmarkType(Enum):
    """Types of benchmarks available"""
    TRAINING_CONVERGENCE = "training_convergence"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    ACCURACY_EVALUATION = "accuracy_evaluation"
    SCALABILITY_ANALYSIS = "scalability_analysis"
    MEMORY_PROFILING = "memory_profiling"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs"""
    # General settings
    benchmark_name: str = "default_benchmark"
    output_directory: str = "benchmark_results"
    save_detailed_logs: bool = True

    # Training benchmarks
    max_training_epochs: int = 100
    problems_per_epoch: int = 20
    convergence_threshold: float = 0.95

    # Accuracy benchmarks
    test_set_sizes: List[int] = field(default_factory=lambda: [100, 500, 1000])
    problem_complexity_ranges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 15),      # Simple: 4-bit problems
        (0, 127),     # Intermediate: 7-bit problems
        (0, 511)      # Advanced: 10-bit problems
    ])

    # Performance sampling
    profiling_interval: float = 1.0  # Seconds between performance samples
    memory_sampling_frequency: int = 10  # Every N epochs

    # Scalability testing
    network_sizes_to_test: List[int] = field(default_factory=lambda: [39, 79, 119])  # Half, full, 1.5x
    parameter_sweep_ranges: Dict[str, List[float]] = field(default_factory=lambda: {
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'problems_per_epoch': [10, 20, 50, 100]
    })


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results"""
    # Timing metrics
    training_time_total: float = 0.0
    training_time_per_epoch: float = 0.0
    inference_time_per_problem: float = 0.0

    # Memory metrics
    peak_memory_usage_mb: float = 0.0
    average_memory_usage_mb: float = 0.0
    memory_growth_rate: float = 0.0

    # CPU metrics
    average_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0

    # Training metrics
    epochs_to_convergence: Optional[int] = None
    final_accuracy: float = 0.0
    convergence_rate: float = 0.0  # accuracy improvement per epoch

    # Network metrics
    network_size_blocks: int = 0
    network_size_wires: int = 0
    parameters_count: int = 0


@dataclass
class BenchmarkResult:
    """Complete result of a benchmark run"""
    benchmark_type: BenchmarkType
    config: BenchmarkConfig
    metrics: PerformanceMetrics

    # Detailed data
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_samples: List[Dict[str, Any]] = field(default_factory=list)
    accuracy_breakdown: Dict[str, float] = field(default_factory=dict)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    duration_seconds: float = 0.0
    success: bool = False
    error_message: Optional[str] = None


class PerformanceBenchmarker:
    """
    Comprehensive performance benchmarking system for Neural-Klotski.

    Measures training convergence, computational efficiency, accuracy
    performance, and scalability characteristics.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmarker.

        Args:
            config: Benchmarking configuration
        """
        self.config = config

        # Create output directory
        self.output_path = Path(config.output_directory)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Performance monitoring
        self.process = psutil.Process()
        self.performance_samples = []

        # Results storage
        self.results: List[BenchmarkResult] = []

    def run_training_convergence_benchmark(self) -> BenchmarkResult:
        """
        Benchmark training convergence characteristics.

        Measures how quickly the network learns addition tasks and
        identifies convergence patterns.
        """
        print("🏃 Running Training Convergence Benchmark...")
        start_time = time.time()

        try:
            # Create network and training configuration
            base_config = get_default_config()
            training_config = TrainingConfig()
            training_config.epochs_per_phase = {
                TrainingPhase.SIMPLE: self.config.max_training_epochs,
                TrainingPhase.INTERMEDIATE: 0,  # Focus on simple for benchmark
                TrainingPhase.ADVANCED: 0
            }
            training_config.problems_per_epoch = self.config.problems_per_epoch
            training_config.evaluation_frequency = 5

            # Create trainer
            trainer = AdditionNetworkTrainer(training_config, base_config.simulation)

            # Performance monitoring
            monitor_config = ConvergenceConfig()
            monitor = PerformanceMonitor(monitor_config)

            # Training history tracking
            training_history = []
            performance_samples = []

            # Training callback for data collection
            def collect_metrics(metrics):
                # Record training metrics
                training_history.append({
                    'epoch': metrics.epoch,
                    'accuracy': metrics.accuracy,
                    'error': metrics.average_error,
                    'confidence': metrics.confidence,
                    'problems_solved': metrics.problems_solved,
                    'total_problems': metrics.total_problems,
                    'timestamp': time.time()
                })

                # Performance monitoring
                snapshot = monitor.record_metrics(metrics)
                convergence_report = monitor.analyze_convergence()

                # System performance sampling
                if len(training_history) % 5 == 0:  # Sample every 5 epochs
                    perf_sample = self._sample_system_performance()
                    perf_sample['epoch'] = metrics.epoch
                    perf_sample['accuracy'] = metrics.accuracy
                    performance_samples.append(perf_sample)

            trainer.add_epoch_callback(collect_metrics)

            # Run training
            print(f"Training network with {len(trainer.network.blocks)} blocks, {len(trainer.network.wires)} wires...")
            final_metrics = trainer.train(max_epochs=self.config.max_training_epochs)

            # Analyze convergence
            epochs_to_convergence = None
            convergence_rate = 0.0

            if training_history:
                # Find convergence point
                for i, record in enumerate(training_history):
                    if record['accuracy'] >= self.config.convergence_threshold:
                        epochs_to_convergence = record['epoch']
                        break

                # Calculate convergence rate (accuracy improvement per epoch)
                if len(training_history) > 10:
                    early_acc = training_history[5]['accuracy']
                    late_acc = training_history[-1]['accuracy']
                    epochs_span = training_history[-1]['epoch'] - training_history[5]['epoch']
                    if epochs_span > 0:
                        convergence_rate = (late_acc - early_acc) / epochs_span

            # Calculate performance metrics
            total_time = time.time() - start_time
            avg_epoch_time = total_time / max(1, final_metrics.epoch) if final_metrics.epoch > 0 else 0

            # Memory analysis
            peak_memory = max((s['memory_mb'] for s in performance_samples), default=0)
            avg_memory = np.mean([s['memory_mb'] for s in performance_samples]) if performance_samples else 0

            # CPU analysis
            avg_cpu = np.mean([s['cpu_percent'] for s in performance_samples]) if performance_samples else 0
            peak_cpu = max((s['cpu_percent'] for s in performance_samples), default=0)

            metrics = PerformanceMetrics(
                training_time_total=total_time,
                training_time_per_epoch=avg_epoch_time,
                epochs_to_convergence=epochs_to_convergence,
                final_accuracy=final_metrics.accuracy,
                convergence_rate=convergence_rate,
                network_size_blocks=len(trainer.network.blocks),
                network_size_wires=len(trainer.network.wires),
                peak_memory_usage_mb=peak_memory,
                average_memory_usage_mb=avg_memory,
                average_cpu_percent=avg_cpu,
                peak_cpu_percent=peak_cpu
            )

            result = BenchmarkResult(
                benchmark_type=BenchmarkType.TRAINING_CONVERGENCE,
                config=self.config,
                metrics=metrics,
                training_history=training_history,
                performance_samples=performance_samples,
                duration_seconds=total_time,
                success=True
            )

            print(f"✅ Training convergence benchmark completed in {total_time:.1f}s")
            print(f"Final accuracy: {final_metrics.accuracy:.3f}")
            if epochs_to_convergence:
                print(f"Converged at epoch {epochs_to_convergence}")

            return result

        except Exception as e:
            print(f"❌ Training convergence benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_type=BenchmarkType.TRAINING_CONVERGENCE,
                config=self.config,
                metrics=PerformanceMetrics(),
                duration_seconds=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def run_accuracy_benchmark(self) -> BenchmarkResult:
        """
        Benchmark accuracy performance across different problem complexities.

        Tests systematic accuracy evaluation on various test sets.
        """
        print("🎯 Running Accuracy Benchmark...")
        start_time = time.time()

        try:
            # Create pre-trained network (simplified for benchmarking)
            config = get_default_config()
            network = create_addition_network(enable_learning=False)
            task_manager = AdditionTaskManager(config.simulation)

            accuracy_breakdown = {}
            all_results = []

            # Test across different complexity ranges
            for i, (min_val, max_val) in enumerate(self.config.problem_complexity_ranges):
                complexity_name = f"range_{min_val}_{max_val}"
                print(f"Testing complexity range: {min_val}-{max_val}")

                # Test across different test set sizes
                for test_size in self.config.test_set_sizes:
                    print(f"  Testing with {test_size} problems...")

                    # Generate test problems
                    test_problems = []
                    for _ in range(test_size):
                        operand1 = np.random.randint(min_val, max_val + 1)
                        operand2 = np.random.randint(min_val, max_val + 1)
                        expected_sum = operand1 + operand2

                        # Skip problems that exceed output capacity
                        if expected_sum < 512:  # 9-bit limit for output
                            test_problems.append(AdditionProblem(operand1, operand2, expected_sum))

                    if not test_problems:
                        continue

                    # Time the accuracy evaluation
                    eval_start = time.time()

                    # Evaluate accuracy
                    correct_count = 0
                    total_error = 0.0
                    total_confidence = 0.0

                    for problem in test_problems:
                        network.reset_simulation()
                        result, stats = task_manager.execute_addition_task(network, problem)

                        is_correct = (result.decoded_sum == problem.expected_sum)
                        if is_correct:
                            correct_count += 1

                        total_error += abs(result.decoded_sum - problem.expected_sum)
                        total_confidence += result.confidence

                    eval_time = time.time() - eval_start

                    # Calculate metrics
                    accuracy = correct_count / len(test_problems)
                    avg_error = total_error / len(test_problems)
                    avg_confidence = total_confidence / len(test_problems)
                    inference_time = eval_time / len(test_problems)

                    # Store results
                    key = f"{complexity_name}_size_{test_size}"
                    accuracy_breakdown[key] = {
                        'accuracy': accuracy,
                        'average_error': avg_error,
                        'confidence': avg_confidence,
                        'inference_time_per_problem': inference_time,
                        'test_size': test_size,
                        'complexity_range': (min_val, max_val)
                    }

                    all_results.append({
                        'complexity_range': (min_val, max_val),
                        'test_size': test_size,
                        'accuracy': accuracy,
                        'inference_time': inference_time
                    })

            # Calculate overall metrics
            if all_results:
                overall_accuracy = np.mean([r['accuracy'] for r in all_results])
                avg_inference_time = np.mean([r['inference_time'] for r in all_results])
            else:
                overall_accuracy = 0.0
                avg_inference_time = 0.0

            total_time = time.time() - start_time

            metrics = PerformanceMetrics(
                final_accuracy=overall_accuracy,
                inference_time_per_problem=avg_inference_time,
                network_size_blocks=len(network.blocks),
                network_size_wires=len(network.wires),
                training_time_total=total_time
            )

            result = BenchmarkResult(
                benchmark_type=BenchmarkType.ACCURACY_EVALUATION,
                config=self.config,
                metrics=metrics,
                accuracy_breakdown=accuracy_breakdown,
                duration_seconds=total_time,
                success=True
            )

            print(f"✅ Accuracy benchmark completed in {total_time:.1f}s")
            print(f"Overall accuracy: {overall_accuracy:.3f}")
            print(f"Average inference time: {avg_inference_time*1000:.2f}ms per problem")

            return result

        except Exception as e:
            print(f"❌ Accuracy benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_type=BenchmarkType.ACCURACY_EVALUATION,
                config=self.config,
                metrics=PerformanceMetrics(),
                duration_seconds=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def run_computational_efficiency_benchmark(self) -> BenchmarkResult:
        """
        Benchmark computational efficiency and resource usage.

        Measures CPU usage, memory consumption, and execution speed.
        """
        print("⚡ Running Computational Efficiency Benchmark...")
        start_time = time.time()

        try:
            # Create network for testing
            config = get_default_config()
            network = create_addition_network(enable_learning=False)
            task_manager = AdditionTaskManager(config.simulation)

            # Performance monitoring setup
            performance_samples = []

            # Memory baseline
            gc.collect()  # Force garbage collection
            baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB

            print(f"Baseline memory: {baseline_memory:.1f} MB")
            print(f"Network: {len(network.blocks)} blocks, {len(network.wires)} wires")

            # CPU and memory intensive operations
            operations_to_test = [
                ("Single Problem Inference", self._benchmark_single_inference),
                ("Batch Problem Processing", self._benchmark_batch_processing),
                ("Network Reset Operations", self._benchmark_network_reset),
                ("Memory Allocation Stress", self._benchmark_memory_stress)
            ]

            operation_results = {}

            for op_name, benchmark_func in operations_to_test:
                print(f"  Testing: {op_name}")

                # Sample performance before operation
                pre_sample = self._sample_system_performance()

                # Run benchmark operation
                op_result = benchmark_func(network, task_manager)

                # Sample performance after operation
                post_sample = self._sample_system_performance()

                # Calculate operation metrics
                operation_results[op_name] = {
                    'duration': op_result.get('duration', 0),
                    'operations_per_second': op_result.get('ops_per_sec', 0),
                    'memory_delta_mb': post_sample['memory_mb'] - pre_sample['memory_mb'],
                    'cpu_peak': max(pre_sample['cpu_percent'], post_sample['cpu_percent']),
                    'details': op_result
                }

                performance_samples.append({
                    'operation': op_name,
                    'timestamp': time.time(),
                    **post_sample
                })

                # Brief pause between operations
                time.sleep(0.5)

            # Calculate overall efficiency metrics
            total_time = time.time() - start_time
            peak_memory = max(s['memory_mb'] for s in performance_samples)
            avg_memory = np.mean([s['memory_mb'] for s in performance_samples])
            memory_growth = peak_memory - baseline_memory

            avg_cpu = np.mean([s['cpu_percent'] for s in performance_samples])
            peak_cpu = max(s['cpu_percent'] for s in performance_samples)

            # Estimate throughput
            single_inference_time = operation_results.get("Single Problem Inference", {}).get('duration', 1.0)
            throughput_problems_per_sec = 1.0 / max(single_inference_time, 1e-6)

            metrics = PerformanceMetrics(
                inference_time_per_problem=single_inference_time,
                peak_memory_usage_mb=peak_memory,
                average_memory_usage_mb=avg_memory,
                memory_growth_rate=memory_growth,
                average_cpu_percent=avg_cpu,
                peak_cpu_percent=peak_cpu,
                network_size_blocks=len(network.blocks),
                network_size_wires=len(network.wires),
                training_time_total=total_time
            )

            result = BenchmarkResult(
                benchmark_type=BenchmarkType.COMPUTATIONAL_EFFICIENCY,
                config=self.config,
                metrics=metrics,
                performance_samples=performance_samples,
                accuracy_breakdown=operation_results,
                duration_seconds=total_time,
                success=True
            )

            print(f"✅ Computational efficiency benchmark completed in {total_time:.1f}s")
            print(f"Peak memory usage: {peak_memory:.1f} MB")
            print(f"Throughput: {throughput_problems_per_sec:.1f} problems/sec")

            return result

        except Exception as e:
            print(f"❌ Computational efficiency benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_type=BenchmarkType.COMPUTATIONAL_EFFICIENCY,
                config=self.config,
                metrics=PerformanceMetrics(),
                duration_seconds=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def _sample_system_performance(self) -> Dict[str, float]:
        """Sample current system performance metrics"""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()

            return {
                'memory_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
                'cpu_percent': cpu_percent,
                'timestamp': time.time()
            }
        except:
            return {
                'memory_mb': 0.0,
                'memory_vms_mb': 0.0,
                'cpu_percent': 0.0,
                'timestamp': time.time()
            }

    def _benchmark_single_inference(self, network, task_manager) -> Dict[str, Any]:
        """Benchmark single problem inference"""
        problem = AdditionProblem(42, 27, 69)

        start_time = time.time()

        # Run single inference
        network.reset_simulation()
        result, stats = task_manager.execute_addition_task(network, problem)

        duration = time.time() - start_time

        return {
            'duration': duration,
            'ops_per_sec': 1.0 / duration if duration > 0 else 0,
            'accuracy': 1.0 if result.decoded_sum == problem.expected_sum else 0.0,
            'confidence': result.confidence
        }

    def _benchmark_batch_processing(self, network, task_manager) -> Dict[str, Any]:
        """Benchmark batch problem processing"""
        batch_size = 50
        problems = [AdditionProblem(i, i+1, 2*i+1) for i in range(batch_size)]

        start_time = time.time()

        correct_count = 0
        for problem in problems:
            network.reset_simulation()
            result, stats = task_manager.execute_addition_task(network, problem)
            if result.decoded_sum == problem.expected_sum:
                correct_count += 1

        duration = time.time() - start_time

        return {
            'duration': duration,
            'ops_per_sec': batch_size / duration if duration > 0 else 0,
            'batch_size': batch_size,
            'accuracy': correct_count / batch_size,
            'avg_time_per_problem': duration / batch_size
        }

    def _benchmark_network_reset(self, network, task_manager) -> Dict[str, Any]:
        """Benchmark network reset operations"""
        num_resets = 100

        start_time = time.time()

        for _ in range(num_resets):
            network.reset_simulation()

        duration = time.time() - start_time

        return {
            'duration': duration,
            'ops_per_sec': num_resets / duration if duration > 0 else 0,
            'num_resets': num_resets,
            'avg_reset_time': duration / num_resets
        }

    def _benchmark_memory_stress(self, network, task_manager) -> Dict[str, Any]:
        """Benchmark memory allocation stress test"""
        # Create multiple network instances to test memory handling
        start_time = time.time()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        networks = []
        for i in range(5):  # Create 5 additional networks
            net = create_addition_network(enable_learning=False)
            networks.append(net)

        peak_memory = self.process.memory_info().rss / 1024 / 1024

        # Clean up
        del networks
        gc.collect()

        final_memory = self.process.memory_info().rss / 1024 / 1024
        duration = time.time() - start_time

        return {
            'duration': duration,
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'final_memory_mb': final_memory,
            'memory_growth_mb': peak_memory - initial_memory,
            'memory_cleanup_mb': peak_memory - final_memory,
            'networks_created': 5
        }

    def run_comprehensive_benchmark(self) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmark suite covering all aspects.

        Returns:
            List of benchmark results
        """
        print("🚀 Running Comprehensive Neural-Klotski Benchmark Suite")
        print("=" * 60)

        benchmarks_to_run = [
            ("Training Convergence", self.run_training_convergence_benchmark),
            ("Accuracy Evaluation", self.run_accuracy_benchmark),
            ("Computational Efficiency", self.run_computational_efficiency_benchmark)
        ]

        results = []

        for benchmark_name, benchmark_func in benchmarks_to_run:
            print(f"\n📊 Starting {benchmark_name} Benchmark...")

            try:
                result = benchmark_func()
                results.append(result)

                if result.success:
                    print(f"✅ {benchmark_name} completed successfully")
                else:
                    print(f"❌ {benchmark_name} failed: {result.error_message}")

            except Exception as e:
                print(f"❌ {benchmark_name} crashed: {e}")
                # Create failed result
                failed_result = BenchmarkResult(
                    benchmark_type=BenchmarkType.COMPUTATIONAL_EFFICIENCY,
                    config=self.config,
                    metrics=PerformanceMetrics(),
                    success=False,
                    error_message=str(e)
                )
                results.append(failed_result)

        # Save results
        self._save_benchmark_results(results)

        # Print summary
        self._print_benchmark_summary(results)

        return results

    def _save_benchmark_results(self, results: List[BenchmarkResult]):
        """Save benchmark results to files"""
        if not self.config.save_detailed_logs:
            return

        timestamp = int(time.time())

        for i, result in enumerate(results):
            filename = f"{result.benchmark_type.value}_{timestamp}_{i}.json"
            filepath = self.output_path / filename

            # Convert result to serializable format
            result_dict = {
                'benchmark_type': result.benchmark_type.value,
                'config': {
                    'benchmark_name': result.config.benchmark_name,
                    'max_training_epochs': result.config.max_training_epochs,
                    'problems_per_epoch': result.config.problems_per_epoch
                },
                'metrics': {
                    'training_time_total': result.metrics.training_time_total,
                    'final_accuracy': result.metrics.final_accuracy,
                    'epochs_to_convergence': result.metrics.epochs_to_convergence,
                    'peak_memory_usage_mb': result.metrics.peak_memory_usage_mb,
                    'average_cpu_percent': result.metrics.average_cpu_percent,
                    'network_size_blocks': result.metrics.network_size_blocks,
                    'network_size_wires': result.metrics.network_size_wires
                },
                'success': result.success,
                'duration_seconds': result.duration_seconds,
                'timestamp': result.timestamp
            }

            try:
                import json
                with open(filepath, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                print(f"📁 Saved {result.benchmark_type.value} results to {filepath}")
            except Exception as e:
                print(f"⚠️  Failed to save {result.benchmark_type.value} results: {e}")

    def _print_benchmark_summary(self, results: List[BenchmarkResult]):
        """Print comprehensive benchmark summary"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE BENCHMARK SUMMARY")
        print("="*60)

        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        print(f"Total benchmarks: {len(results)}")
        print(f"Successful: {len(successful_results)} ✅")
        print(f"Failed: {len(failed_results)} ❌")

        if successful_results:
            print(f"\n🎯 Performance Summary:")

            for result in successful_results:
                print(f"\n{result.benchmark_type.value.replace('_', ' ').title()}:")
                m = result.metrics

                if result.benchmark_type == BenchmarkType.TRAINING_CONVERGENCE:
                    print(f"  • Final accuracy: {m.final_accuracy:.3f}")
                    if m.epochs_to_convergence:
                        print(f"  • Epochs to convergence: {m.epochs_to_convergence}")
                    print(f"  • Training time: {m.training_time_total:.1f}s")
                    print(f"  • Peak memory: {m.peak_memory_usage_mb:.1f} MB")

                elif result.benchmark_type == BenchmarkType.ACCURACY_EVALUATION:
                    print(f"  • Overall accuracy: {m.final_accuracy:.3f}")
                    print(f"  • Avg inference time: {m.inference_time_per_problem*1000:.2f}ms")

                elif result.benchmark_type == BenchmarkType.COMPUTATIONAL_EFFICIENCY:
                    print(f"  • Peak memory usage: {m.peak_memory_usage_mb:.1f} MB")
                    print(f"  • Average CPU usage: {m.average_cpu_percent:.1f}%")
                    print(f"  • Inference time: {m.inference_time_per_problem*1000:.2f}ms per problem")

        if failed_results:
            print(f"\n❌ Failed Benchmarks:")
            for result in failed_results:
                print(f"  • {result.benchmark_type.value}: {result.error_message}")

        print(f"\n📁 Results saved to: {self.output_path}")
        print("="*60)


if __name__ == "__main__":
    # Run comprehensive benchmark
    print("🚀 Neural-Klotski Performance Benchmarking")

    config = BenchmarkConfig(
        benchmark_name="comprehensive_test",
        max_training_epochs=20,  # Shorter for testing
        problems_per_epoch=10,
        test_set_sizes=[50, 100],  # Smaller test sets
        output_directory="test_benchmarks"
    )

    benchmarker = PerformanceBenchmarker(config)
    results = benchmarker.run_comprehensive_benchmark()

    print(f"\n🎉 Benchmarking completed! {len([r for r in results if r.success])} successful benchmarks.")