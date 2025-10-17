"""
Computational Efficiency Profiler for Neural-Klotski System.

Provides detailed profiling of computational performance, memory usage,
and execution characteristics for optimization analysis.
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import time
import psutil
import gc
import tracemalloc
import cProfile
import pstats
from io import StringIO
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
class ProfiledOperation:
    """Container for profiled operation results"""
    operation_name: str
    duration_seconds: float
    cpu_time_seconds: float
    memory_peak_mb: float
    memory_delta_mb: float
    calls_count: int
    operations_per_second: float

    # Detailed metrics
    function_profile: Optional[str] = None
    memory_trace: List[Tuple[float, float]] = field(default_factory=list)  # (time, memory_mb)
    cpu_samples: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"ProfiledOperation({self.operation_name}: "
                f"{self.duration_seconds:.3f}s, "
                f"{self.memory_peak_mb:.1f}MB peak, "
                f"{self.operations_per_second:.1f} ops/sec)")


@dataclass
class SystemProfile:
    """Complete system performance profile"""
    # Overall metrics
    total_duration: float = 0.0
    peak_memory_usage_mb: float = 0.0
    average_cpu_percent: float = 0.0

    # Component profiles
    operation_profiles: Dict[str, ProfiledOperation] = field(default_factory=dict)

    # System characteristics
    system_info: Dict[str, Any] = field(default_factory=dict)
    python_version: str = ""
    platform_info: str = ""

    # Performance bottlenecks
    slowest_operations: List[Tuple[str, float]] = field(default_factory=list)
    memory_hotspots: List[Tuple[str, float]] = field(default_factory=list)


class EfficiencyProfiler:
    """
    Detailed computational efficiency profiler for Neural-Klotski.

    Provides fine-grained analysis of performance characteristics,
    memory usage patterns, and computational bottlenecks.
    """

    def __init__(self, enable_detailed_profiling: bool = True):
        """
        Initialize profiler.

        Args:
            enable_detailed_profiling: Enable detailed function-level profiling
        """
        self.enable_detailed_profiling = enable_detailed_profiling
        self.process = psutil.Process()

        # Profiling state
        self.profiled_operations: Dict[str, ProfiledOperation] = {}
        self.system_samples: List[Dict[str, float]] = []

        # System information
        self.system_info = self._gather_system_info()

    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather system information for profiling context"""
        import platform

        try:
            return {
                'cpu_count': psutil.cpu_count(logical=True),
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': platform.python_version(),
                'platform': platform.platform(),
                'architecture': platform.architecture()[0]
            }
        except:
            return {'error': 'Could not gather system info'}

    def profile_operation(self, operation_name: str, operation_func: Callable, *args, **kwargs) -> ProfiledOperation:
        """
        Profile a specific operation with detailed metrics.

        Args:
            operation_name: Name of the operation
            operation_func: Function to profile
            *args, **kwargs: Arguments for the function

        Returns:
            Detailed profiling results
        """
        print(f"🔍 Profiling: {operation_name}")

        # Memory tracking setup
        if self.enable_detailed_profiling:
            tracemalloc.start()

        # Baseline measurements
        gc.collect()  # Force garbage collection
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        initial_time = time.time()

        # CPU profiling setup
        profiler = None
        if self.enable_detailed_profiling:
            profiler = cProfile.Profile()
            profiler.enable()

        # Memory and CPU sampling during execution
        memory_trace = [(0.0, initial_memory)]
        cpu_samples = []

        try:
            # Execute the operation
            start_time = time.time()
            result = operation_func(*args, **kwargs)
            end_time = time.time()

            duration = end_time - start_time

            # Sample final performance
            final_memory = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()

            memory_trace.append((duration, final_memory))
            cpu_samples.append(cpu_percent)

            # Detailed profiling results
            function_profile_text = None
            if self.enable_detailed_profiling and profiler:
                profiler.disable()

                # Capture profiling stats
                stats_buffer = StringIO()
                stats = pstats.Stats(profiler, stream=stats_buffer)
                stats.sort_stats('cumulative')
                stats.print_stats(20)  # Top 20 functions
                function_profile_text = stats_buffer.getvalue()

            # Memory profiling results
            peak_memory = max(sample[1] for sample in memory_trace)
            memory_delta = final_memory - initial_memory

            # Calculate operations per second (assume 1 operation if not specified)
            operations_count = kwargs.get('operations_count', 1)
            ops_per_sec = operations_count / duration if duration > 0 else 0

            # Create profiled operation
            profiled_op = ProfiledOperation(
                operation_name=operation_name,
                duration_seconds=duration,
                cpu_time_seconds=duration,  # Approximation
                memory_peak_mb=peak_memory,
                memory_delta_mb=memory_delta,
                calls_count=operations_count,
                operations_per_second=ops_per_sec,
                function_profile=function_profile_text,
                memory_trace=memory_trace,
                cpu_samples=cpu_samples
            )

            print(f"  ✅ {operation_name}: {duration:.3f}s, {peak_memory:.1f}MB peak, {ops_per_sec:.1f} ops/sec")

            # Store for analysis
            self.profiled_operations[operation_name] = profiled_op

            return profiled_op

        except Exception as e:
            print(f"  ❌ {operation_name} failed: {e}")
            # Return failed operation profile
            return ProfiledOperation(
                operation_name=f"{operation_name}_FAILED",
                duration_seconds=time.time() - initial_time,
                cpu_time_seconds=0,
                memory_peak_mb=initial_memory,
                memory_delta_mb=0,
                calls_count=0,
                operations_per_second=0
            )
        finally:
            if self.enable_detailed_profiling and tracemalloc.is_tracing():
                tracemalloc.stop()

    def profile_network_creation(self) -> ProfiledOperation:
        """Profile network creation performance"""
        def create_network():
            return create_addition_network(enable_learning=True)

        return self.profile_operation("Network Creation", create_network)

    def profile_single_inference(self, network=None) -> ProfiledOperation:
        """Profile single problem inference"""
        if network is None:
            network = create_addition_network(enable_learning=False)

        config = get_default_config()
        task_manager = AdditionTaskManager(config.simulation)
        problem = AdditionProblem(42, 27, 69)

        def run_inference():
            network.reset_simulation()
            return task_manager.execute_addition_task(network, problem)

        return self.profile_operation("Single Inference", run_inference)

    def profile_batch_inference(self, batch_size: int = 100, network=None) -> ProfiledOperation:
        """Profile batch inference performance"""
        if network is None:
            network = create_addition_network(enable_learning=False)

        config = get_default_config()
        task_manager = AdditionTaskManager(config.simulation)
        problems = [AdditionProblem(i % 100, (i+1) % 100, (2*i+1) % 100) for i in range(batch_size)]

        def run_batch_inference():
            results = []
            for problem in problems:
                network.reset_simulation()
                result, stats = task_manager.execute_addition_task(network, problem)
                results.append(result)
            return results

        return self.profile_operation(
            f"Batch Inference ({batch_size})",
            run_batch_inference,
            operations_count=batch_size
        )

    def profile_training_epoch(self) -> ProfiledOperation:
        """Profile single training epoch"""
        config = get_default_config()
        training_config = TrainingConfig()
        training_config.epochs_per_phase = {
            TrainingPhase.SIMPLE: 1,
            TrainingPhase.INTERMEDIATE: 0,
            TrainingPhase.ADVANCED: 0
        }
        training_config.problems_per_epoch = 10

        def run_training_epoch():
            trainer = AdditionNetworkTrainer(training_config, config.simulation)
            return trainer.train(max_epochs=1)

        return self.profile_operation("Training Epoch", run_training_epoch, operations_count=10)

    def profile_network_reset(self, num_resets: int = 1000, network=None) -> ProfiledOperation:
        """Profile network reset operations"""
        if network is None:
            network = create_addition_network(enable_learning=False)

        def run_resets():
            for _ in range(num_resets):
                network.reset_simulation()

        return self.profile_operation(
            f"Network Reset ({num_resets})",
            run_resets,
            operations_count=num_resets
        )

    def profile_memory_scaling(self) -> ProfiledOperation:
        """Profile memory scaling with multiple networks"""
        def create_multiple_networks():
            networks = []
            for i in range(10):  # Create 10 networks
                net = create_addition_network(enable_learning=False)
                networks.append(net)
            return networks

        return self.profile_operation("Memory Scaling (10 networks)", create_multiple_networks, operations_count=10)

    def run_comprehensive_profiling(self) -> SystemProfile:
        """
        Run comprehensive efficiency profiling across all operations.

        Returns:
            Complete system performance profile
        """
        print("🔬 Running Comprehensive Efficiency Profiling")
        print("=" * 60)

        overall_start = time.time()

        # Profile core operations
        operations_to_profile = [
            ("Network Creation", self.profile_network_creation),
            ("Single Inference", self.profile_single_inference),
            ("Batch Inference (50)", lambda: self.profile_batch_inference(50)),
            ("Training Epoch", self.profile_training_epoch),
            ("Network Reset (500)", lambda: self.profile_network_reset(500)),
            ("Memory Scaling", self.profile_memory_scaling)
        ]

        # Run profiling
        for op_name, profile_func in operations_to_profile:
            try:
                profile_func()
            except Exception as e:
                print(f"❌ Failed to profile {op_name}: {e}")

        overall_duration = time.time() - overall_start

        # Analyze results
        profile = self._analyze_profiling_results(overall_duration)

        # Print summary
        self._print_profiling_summary(profile)

        return profile

    def _analyze_profiling_results(self, total_duration: float) -> SystemProfile:
        """Analyze profiling results and identify bottlenecks"""
        if not self.profiled_operations:
            return SystemProfile()

        # Calculate overall metrics
        peak_memory = max(op.memory_peak_mb for op in self.profiled_operations.values())
        avg_ops_per_sec = np.mean([op.operations_per_second for op in self.profiled_operations.values()
                                  if op.operations_per_second > 0])

        # Identify bottlenecks
        slowest_ops = sorted(
            [(name, op.duration_seconds) for name, op in self.profiled_operations.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        memory_hotspots = sorted(
            [(name, op.memory_peak_mb) for name, op in self.profiled_operations.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return SystemProfile(
            total_duration=total_duration,
            peak_memory_usage_mb=peak_memory,
            average_cpu_percent=avg_ops_per_sec,  # Approximate
            operation_profiles=self.profiled_operations,
            system_info=self.system_info,
            slowest_operations=slowest_ops,
            memory_hotspots=memory_hotspots
        )

    def _print_profiling_summary(self, profile: SystemProfile):
        """Print comprehensive profiling summary"""
        print("\n" + "="*60)
        print("🔬 EFFICIENCY PROFILING SUMMARY")
        print("="*60)

        print(f"Total profiling time: {profile.total_duration:.1f}s")
        print(f"Peak memory usage: {profile.peak_memory_usage_mb:.1f} MB")

        # System information
        print(f"\n💻 System Information:")
        for key, value in profile.system_info.items():
            print(f"  • {key}: {value}")

        # Operation performance
        print(f"\n⚡ Operation Performance:")
        for name, op in profile.operation_profiles.items():
            if op.operations_per_second > 0:
                print(f"  • {name}:")
                print(f"    - Duration: {op.duration_seconds:.3f}s")
                print(f"    - Throughput: {op.operations_per_second:.1f} ops/sec")
                print(f"    - Peak memory: {op.memory_peak_mb:.1f} MB")
                print(f"    - Memory delta: {op.memory_delta_mb:+.1f} MB")

        # Performance bottlenecks
        if profile.slowest_operations:
            print(f"\n🐌 Slowest Operations:")
            for name, duration in profile.slowest_operations:
                print(f"  • {name}: {duration:.3f}s")

        if profile.memory_hotspots:
            print(f"\n🔥 Memory Hotspots:")
            for name, memory in profile.memory_hotspots:
                print(f"  • {name}: {memory:.1f} MB")

        # Performance recommendations
        print(f"\n💡 Performance Recommendations:")
        self._generate_performance_recommendations(profile)

        print("="*60)

    def _generate_performance_recommendations(self, profile: SystemProfile):
        """Generate performance optimization recommendations"""
        recommendations = []

        # Check for slow operations
        if profile.slowest_operations:
            slowest_name, slowest_time = profile.slowest_operations[0]
            if slowest_time > 5.0:
                recommendations.append(f"Optimize '{slowest_name}' - taking {slowest_time:.1f}s")

        # Check for memory usage
        if profile.peak_memory_usage_mb > 500:
            recommendations.append(f"High memory usage detected ({profile.peak_memory_usage_mb:.1f} MB)")

        # Check for inefficient operations
        for name, op in profile.operation_profiles.items():
            if "Inference" in name and op.operations_per_second < 10:
                recommendations.append(f"Low inference throughput in '{name}' ({op.operations_per_second:.1f} ops/sec)")

        # Memory growth check
        for name, op in profile.operation_profiles.items():
            if op.memory_delta_mb > 50:
                recommendations.append(f"Potential memory leak in '{name}' (+{op.memory_delta_mb:.1f} MB)")

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("  ✅ No major performance issues detected")

    def get_detailed_function_profile(self, operation_name: str) -> Optional[str]:
        """Get detailed function-level profiling for an operation"""
        if operation_name in self.profiled_operations:
            return self.profiled_operations[operation_name].function_profile
        return None

    def export_profiling_data(self, filename: str = "efficiency_profile.json"):
        """Export profiling data to JSON file"""
        try:
            import json

            export_data = {
                'system_info': self.system_info,
                'operations': {}
            }

            for name, op in self.profiled_operations.items():
                export_data['operations'][name] = {
                    'duration_seconds': op.duration_seconds,
                    'memory_peak_mb': op.memory_peak_mb,
                    'memory_delta_mb': op.memory_delta_mb,
                    'operations_per_second': op.operations_per_second,
                    'calls_count': op.calls_count
                }

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)

            print(f"📁 Profiling data exported to {filename}")

        except Exception as e:
            print(f"❌ Failed to export profiling data: {e}")


if __name__ == "__main__":
    # Test efficiency profiler
    print("🔬 Testing Neural-Klotski Efficiency Profiler...")

    profiler = EfficiencyProfiler(enable_detailed_profiling=True)

    # Run comprehensive profiling
    profile = profiler.run_comprehensive_profiling()

    # Export results
    profiler.export_profiling_data("test_efficiency_profile.json")

    print("\n✅ Efficiency profiling test completed!")