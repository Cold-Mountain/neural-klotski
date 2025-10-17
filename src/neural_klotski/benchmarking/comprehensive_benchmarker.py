"""
Comprehensive Benchmarking Suite for Neural-Klotski System.

Integrates all benchmarking components to provide complete performance analysis:
training convergence, computational efficiency, accuracy evaluation, and scalability testing.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from .performance_benchmarker import PerformanceBenchmarker, ComprehensiveBenchmarkReport
from .efficiency_profiler import EfficiencyProfiler, SystemProfile
from .accuracy_benchmarker import AccuracyBenchmarker, ComprehensiveAccuracyReport
from .scalability_tester import ScalabilityTester, ComprehensiveScalabilityReport


@dataclass
class CompleteBenchmarkReport:
    """Complete benchmarking analysis across all dimensions"""
    # Individual component reports
    performance_report: ComprehensiveBenchmarkReport
    efficiency_report: SystemProfile
    accuracy_report: ComprehensiveAccuracyReport
    scalability_report: ComprehensiveScalabilityReport

    # Integrated analysis
    overall_score: float
    system_readiness: str  # "research", "development", "production"
    key_findings: List[str]
    priority_recommendations: List[str]

    # Benchmark metadata
    benchmark_duration: float
    timestamp: str
    system_configuration: Dict[str, Any]


class ComprehensiveBenchmarker:
    """
    Complete benchmarking suite for Neural-Klotski system.

    Provides integrated performance analysis across all dimensions:
    - Training convergence and performance metrics
    - Computational efficiency and resource usage
    - Accuracy evaluation across problem complexities
    - Scalability characteristics and limits

    Generates actionable insights and recommendations for optimization.
    """

    def __init__(self, enable_full_analysis: bool = True):
        """
        Initialize comprehensive benchmarker.

        Args:
            enable_full_analysis: Enable all benchmarking components (may take longer)
        """
        self.enable_full_analysis = enable_full_analysis

        # Initialize individual benchmarking components
        self.performance_benchmarker = PerformanceBenchmarker()
        self.efficiency_profiler = EfficiencyProfiler(enable_detailed_profiling=True)
        self.accuracy_benchmarker = AccuracyBenchmarker()
        self.scalability_tester = ScalabilityTester()

    def run_complete_benchmark_suite(self) -> CompleteBenchmarkReport:
        """
        Run complete benchmarking suite across all analysis dimensions.

        Returns:
            Comprehensive benchmark report with integrated analysis
        """
        print("🔬 NEURAL-KLOTSKI COMPREHENSIVE BENCHMARKING SUITE")
        print("=" * 80)
        print("Running complete performance analysis across all dimensions...")
        print("This may take several minutes to complete.\n")

        overall_start = time.time()

        # Initialize reports
        performance_report = None
        efficiency_report = None
        accuracy_report = None
        scalability_report = None

        # 1. Performance Benchmarking (Training convergence, computational performance)
        print("📊 Phase 1: Performance Benchmarking")
        print("-" * 40)
        try:
            performance_report = self.performance_benchmarker.run_comprehensive_benchmark()
            print("✅ Performance benchmarking completed\n")
        except Exception as e:
            print(f"❌ Performance benchmarking failed: {e}\n")

        # 2. Efficiency Profiling (Memory, CPU, computational bottlenecks)
        print("⚡ Phase 2: Efficiency Profiling")
        print("-" * 40)
        try:
            efficiency_report = self.efficiency_profiler.run_comprehensive_profiling()
            print("✅ Efficiency profiling completed\n")
        except Exception as e:
            print(f"❌ Efficiency profiling failed: {e}\n")

        # 3. Accuracy Evaluation (Systematic test problems)
        print("🎯 Phase 3: Accuracy Evaluation")
        print("-" * 40)
        try:
            accuracy_report = self.accuracy_benchmarker.run_comprehensive_accuracy_evaluation()
            print("✅ Accuracy evaluation completed\n")
        except Exception as e:
            print(f"❌ Accuracy evaluation failed: {e}\n")

        # 4. Scalability Testing (Resource scaling, performance limits)
        if self.enable_full_analysis:
            print("📈 Phase 4: Scalability Testing")
            print("-" * 40)
            try:
                scalability_report = self.scalability_tester.run_comprehensive_scalability_analysis()
                print("✅ Scalability testing completed\n")
            except Exception as e:
                print(f"❌ Scalability testing failed: {e}\n")

        benchmark_duration = time.time() - overall_start

        # Generate integrated analysis
        complete_report = self._generate_integrated_analysis(
            performance_report,
            efficiency_report,
            accuracy_report,
            scalability_report,
            benchmark_duration
        )

        # Print comprehensive summary
        self._print_complete_summary(complete_report)

        return complete_report

    def run_quick_benchmark(self) -> CompleteBenchmarkReport:
        """
        Run abbreviated benchmarking for faster analysis.

        Focuses on core performance metrics without extensive scalability testing.
        """
        print("⚡ NEURAL-KLOTSKI QUICK BENCHMARK")
        print("=" * 50)
        print("Running abbreviated performance analysis...\n")

        overall_start = time.time()

        # Quick performance check
        print("📊 Quick Performance Check")
        print("-" * 30)
        performance_report = None
        try:
            # Run abbreviated performance benchmark
            performance_report = self.performance_benchmarker.run_training_convergence_benchmark()
            print("✅ Performance check completed\n")
        except Exception as e:
            print(f"❌ Performance check failed: {e}\n")

        # Essential efficiency profiling
        print("⚡ Essential Efficiency Profiling")
        print("-" * 30)
        efficiency_report = None
        try:
            # Run core efficiency operations only
            profiler = EfficiencyProfiler(enable_detailed_profiling=False)
            profiler.profile_network_creation()
            profiler.profile_single_inference()
            profiler.profile_batch_inference(25)
            efficiency_report = profiler._analyze_profiling_results(time.time() - overall_start)
            print("✅ Efficiency profiling completed\n")
        except Exception as e:
            print(f"❌ Efficiency profiling failed: {e}\n")

        # Limited accuracy evaluation
        print("🎯 Core Accuracy Evaluation")
        print("-" * 30)
        accuracy_report = None
        try:
            # Test only simple and intermediate ranges
            benchmarker = AccuracyBenchmarker()
            benchmarker.test_suites = [suite for suite in benchmarker.test_suites
                                     if suite.name in ["simple_range", "intermediate_range"]]
            accuracy_report = benchmarker.run_comprehensive_accuracy_evaluation()
            print("✅ Accuracy evaluation completed\n")
        except Exception as e:
            print(f"❌ Accuracy evaluation failed: {e}\n")

        benchmark_duration = time.time() - overall_start

        # Generate integrated analysis (no scalability)
        complete_report = self._generate_integrated_analysis(
            performance_report,
            efficiency_report,
            accuracy_report,
            None,  # No scalability testing
            benchmark_duration
        )

        # Print summary
        self._print_complete_summary(complete_report)

        return complete_report

    def _generate_integrated_analysis(self,
                                    performance_report: Optional[ComprehensiveBenchmarkReport],
                                    efficiency_report: Optional[SystemProfile],
                                    accuracy_report: Optional[ComprehensiveAccuracyReport],
                                    scalability_report: Optional[ComprehensiveScalabilityReport],
                                    benchmark_duration: float) -> CompleteBenchmarkReport:
        """Generate integrated analysis from component reports"""

        # Calculate overall score (0-100)
        overall_score = self._calculate_overall_score(
            performance_report, efficiency_report, accuracy_report, scalability_report
        )

        # Determine system readiness
        system_readiness = self._assess_system_readiness(overall_score, accuracy_report)

        # Generate key findings
        key_findings = self._extract_key_findings(
            performance_report, efficiency_report, accuracy_report, scalability_report
        )

        # Generate priority recommendations
        priority_recommendations = self._generate_priority_recommendations(
            performance_report, efficiency_report, accuracy_report, scalability_report
        )

        # Gather system configuration
        system_config = self._gather_system_configuration()

        return CompleteBenchmarkReport(
            performance_report=performance_report,
            efficiency_report=efficiency_report,
            accuracy_report=accuracy_report,
            scalability_report=scalability_report,
            overall_score=overall_score,
            system_readiness=system_readiness,
            key_findings=key_findings,
            priority_recommendations=priority_recommendations,
            benchmark_duration=benchmark_duration,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_configuration=system_config
        )

    def _calculate_overall_score(self,
                               performance_report: Optional[ComprehensiveBenchmarkReport],
                               efficiency_report: Optional[SystemProfile],
                               accuracy_report: Optional[ComprehensiveAccuracyReport],
                               scalability_report: Optional[ComprehensiveScalabilityReport]) -> float:
        """Calculate overall system score (0-100)"""
        score_components = []

        # Performance score (0-25 points)
        if performance_report and performance_report.training_performance:
            training_acc = performance_report.training_performance.final_accuracy
            performance_score = min(25, training_acc * 25)
            score_components.append(performance_score)

        # Efficiency score (0-25 points)
        if efficiency_report and efficiency_report.operation_profiles:
            # Score based on reasonable performance
            inference_ops = [op for name, op in efficiency_report.operation_profiles.items()
                           if "Inference" in name]
            if inference_ops:
                avg_ops_per_sec = sum(op.operations_per_second for op in inference_ops) / len(inference_ops)
                efficiency_score = min(25, (avg_ops_per_sec / 100) * 25)  # 100 ops/sec = full score
                score_components.append(efficiency_score)

        # Accuracy score (0-30 points)
        if accuracy_report:
            accuracy_score = accuracy_report.overall_accuracy * 30
            score_components.append(accuracy_score)

        # Scalability score (0-20 points)
        if scalability_report and scalability_report.test_results:
            # Score based on lack of major bottlenecks
            total_bottlenecks = len(scalability_report.primary_bottlenecks)
            scalability_score = max(0, 20 - (total_bottlenecks * 5))
            score_components.append(scalability_score)

        # Calculate weighted average
        if score_components:
            return sum(score_components) / len(score_components) * (100 / 25)  # Normalize to 100
        else:
            return 0.0

    def _assess_system_readiness(self, overall_score: float, accuracy_report: Optional[ComprehensiveAccuracyReport]) -> str:
        """Assess system readiness level"""
        if overall_score >= 80 and accuracy_report and accuracy_report.overall_accuracy >= 0.9:
            return "production"
        elif overall_score >= 60 and accuracy_report and accuracy_report.overall_accuracy >= 0.7:
            return "development"
        else:
            return "research"

    def _extract_key_findings(self,
                            performance_report: Optional[ComprehensiveBenchmarkReport],
                            efficiency_report: Optional[SystemProfile],
                            accuracy_report: Optional[ComprehensiveAccuracyReport],
                            scalability_report: Optional[ComprehensiveScalabilityReport]) -> List[str]:
        """Extract key findings from component reports"""
        findings = []

        # Performance findings
        if performance_report and performance_report.training_performance:
            acc = performance_report.training_performance.final_accuracy
            findings.append(f"Training achieves {acc:.1%} accuracy in {performance_report.training_performance.total_epochs} epochs")

        # Efficiency findings
        if efficiency_report and efficiency_report.peak_memory_usage_mb:
            findings.append(f"Peak memory usage: {efficiency_report.peak_memory_usage_mb:.1f} MB")

        # Accuracy findings
        if accuracy_report:
            findings.append(f"Overall accuracy across test suites: {accuracy_report.overall_accuracy:.1%}")
            if accuracy_report.best_performing_suites:
                best_suite = accuracy_report.best_performing_suites[0]
                findings.append(f"Best performance on {best_suite[0]} suite: {best_suite[1]:.1%}")

        # Scalability findings
        if scalability_report and scalability_report.primary_bottlenecks:
            findings.append(f"Primary scaling bottleneck: {scalability_report.primary_bottlenecks[0]}")

        # System behavior
        if efficiency_report and efficiency_report.operation_profiles:
            fastest_op = min(efficiency_report.operation_profiles.values(),
                           key=lambda op: op.duration_seconds)
            findings.append(f"Fastest operation: {fastest_op.operation_name} ({fastest_op.duration_seconds:.3f}s)")

        return findings

    def _generate_priority_recommendations(self,
                                         performance_report: Optional[ComprehensiveBenchmarkReport],
                                         efficiency_report: Optional[SystemProfile],
                                         accuracy_report: Optional[ComprehensiveAccuracyReport],
                                         scalability_report: Optional[ComprehensiveScalabilityReport]) -> List[str]:
        """Generate priority recommendations for optimization"""
        recommendations = []

        # Performance recommendations
        if performance_report and performance_report.optimization_recommendations:
            recommendations.extend(performance_report.optimization_recommendations[:2])

        # Efficiency recommendations
        if efficiency_report and efficiency_report.operation_profiles:
            slowest_ops = efficiency_report.slowest_operations
            if slowest_ops:
                slowest_name, slowest_time = slowest_ops[0]
                if slowest_time > 5.0:
                    recommendations.append(f"Optimize {slowest_name} performance (currently {slowest_time:.1f}s)")

        # Accuracy recommendations
        if accuracy_report and accuracy_report.training_recommendations:
            recommendations.extend(accuracy_report.training_recommendations[:2])

        # Scalability recommendations
        if scalability_report and scalability_report.resource_recommendations:
            recommendations.extend(scalability_report.resource_recommendations[:2])

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations[:5]  # Top 5 priorities

    def _gather_system_configuration(self) -> Dict[str, Any]:
        """Gather system configuration information"""
        try:
            import platform
            import psutil

            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(logical=True),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'neural_klotski_version': "1.0.0"  # TODO: Get from package
            }
        except:
            return {'error': 'Could not gather system configuration'}

    def _print_complete_summary(self, report: CompleteBenchmarkReport):
        """Print comprehensive benchmark summary"""
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE BENCHMARK SUMMARY")
        print("="*80)

        # Overall assessment
        print(f"Overall Score: {report.overall_score:.1f}/100")
        print(f"System Readiness: {report.system_readiness.upper()}")
        print(f"Benchmark Duration: {report.benchmark_duration:.1f}s")
        print(f"Timestamp: {report.timestamp}")

        # System configuration
        if report.system_configuration:
            print(f"\n💻 System Configuration:")
            for key, value in report.system_configuration.items():
                print(f"  • {key}: {value}")

        # Component summaries
        if report.performance_report:
            print(f"\n📊 Performance Summary:")
            if report.performance_report.training_performance:
                perf = report.performance_report.training_performance
                print(f"  • Training accuracy: {perf.final_accuracy:.1%}")
                print(f"  • Training epochs: {perf.total_epochs}")
                print(f"  • Average confidence: {perf.average_confidence:.3f}")

        if report.efficiency_report:
            print(f"\n⚡ Efficiency Summary:")
            print(f"  • Peak memory: {report.efficiency_report.peak_memory_usage_mb:.1f} MB")
            if report.efficiency_report.operation_profiles:
                fastest = min(report.efficiency_report.operation_profiles.values(),
                            key=lambda op: op.duration_seconds)
                print(f"  • Fastest operation: {fastest.operation_name} ({fastest.duration_seconds:.3f}s)")

        if report.accuracy_report:
            print(f"\n🎯 Accuracy Summary:")
            print(f"  • Overall accuracy: {report.accuracy_report.overall_accuracy:.1%}")
            print(f"  • Test suites completed: {len(report.accuracy_report.suite_results)}")

        if report.scalability_report:
            print(f"\n📈 Scalability Summary:")
            print(f"  • Tests completed: {len(report.scalability_report.test_results)}")
            if report.scalability_report.primary_bottlenecks:
                print(f"  • Primary bottleneck: {report.scalability_report.primary_bottlenecks[0]}")

        # Key findings
        if report.key_findings:
            print(f"\n🔍 Key Findings:")
            for i, finding in enumerate(report.key_findings, 1):
                print(f"  {i}. {finding}")

        # Priority recommendations
        if report.priority_recommendations:
            print(f"\n💡 Priority Recommendations:")
            for i, rec in enumerate(report.priority_recommendations, 1):
                print(f"  {i}. {rec}")

        # Readiness assessment
        print(f"\n🚀 System Readiness Assessment:")
        if report.system_readiness == "production":
            print("  ✅ PRODUCTION READY - System performs well across all benchmarks")
        elif report.system_readiness == "development":
            print("  🔧 DEVELOPMENT READY - Good performance with some optimization opportunities")
        else:
            print("  🔬 RESEARCH PHASE - Significant improvements needed before deployment")

        print("="*80)

    def export_complete_results(self, report: CompleteBenchmarkReport, filename: str = "complete_benchmark.json"):
        """Export complete benchmark results to JSON"""
        try:
            export_data = {
                'overall_score': report.overall_score,
                'system_readiness': report.system_readiness,
                'key_findings': report.key_findings,
                'priority_recommendations': report.priority_recommendations,
                'benchmark_duration': report.benchmark_duration,
                'timestamp': report.timestamp,
                'system_configuration': report.system_configuration,
                'component_summaries': {}
            }

            # Add component summaries (simplified)
            if report.performance_report and report.performance_report.training_performance:
                export_data['component_summaries']['performance'] = {
                    'final_accuracy': report.performance_report.training_performance.final_accuracy,
                    'total_epochs': report.performance_report.training_performance.total_epochs,
                    'convergence_achieved': report.performance_report.training_performance.convergence_achieved
                }

            if report.efficiency_report:
                export_data['component_summaries']['efficiency'] = {
                    'peak_memory_mb': report.efficiency_report.peak_memory_usage_mb,
                    'total_duration': report.efficiency_report.total_duration
                }

            if report.accuracy_report:
                export_data['component_summaries']['accuracy'] = {
                    'overall_accuracy': report.accuracy_report.overall_accuracy,
                    'suites_completed': len(report.accuracy_report.suite_results)
                }

            if report.scalability_report:
                export_data['component_summaries']['scalability'] = {
                    'tests_completed': len(report.scalability_report.test_results),
                    'primary_bottlenecks': report.scalability_report.primary_bottlenecks
                }

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)

            print(f"📁 Complete benchmark results exported to {filename}")

        except Exception as e:
            print(f"❌ Failed to export complete results: {e}")


if __name__ == "__main__":
    # Test comprehensive benchmarker
    print("🔬 Testing Neural-Klotski Comprehensive Benchmarker...")

    # Run quick benchmark for testing
    benchmarker = ComprehensiveBenchmarker(enable_full_analysis=False)
    report = benchmarker.run_quick_benchmark()

    # Export results
    benchmarker.export_complete_results(report, "test_complete_benchmark.json")

    print("\n✅ Comprehensive benchmarking test completed!")