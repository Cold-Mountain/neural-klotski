"""
Accuracy Benchmarking System for Neural-Klotski.

Provides systematic accuracy evaluation across different problem complexities,
number ranges, and testing scenarios to validate system performance.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
import random
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
class AccuracyTestSuite:
    """Configuration for systematic accuracy testing"""
    name: str
    description: str
    min_operand: int = 0
    max_operand: int = 511
    num_problems: int = 100
    include_edge_cases: bool = True
    problem_distribution: str = "uniform"  # "uniform", "gaussian", "edge_heavy"


@dataclass
class ProblemResult:
    """Result for a single addition problem test"""
    problem: AdditionProblem
    network_output: int
    correct: bool
    confidence: float
    execution_time: float
    error_magnitude: int = 0  # |expected - actual|

    def __post_init__(self):
        if not self.correct:
            self.error_magnitude = abs(self.problem.expected_sum - self.network_output)


@dataclass
class SuiteResult:
    """Results for an entire test suite"""
    suite_name: str
    total_problems: int
    correct_answers: int
    accuracy: float
    average_confidence: float
    average_execution_time: float

    # Error analysis
    error_distribution: Dict[int, int] = field(default_factory=dict)  # error_magnitude -> count
    worst_errors: List[ProblemResult] = field(default_factory=list)  # Top 10 worst errors

    # Performance analysis
    fast_problems: List[ProblemResult] = field(default_factory=list)  # Fastest 10%
    slow_problems: List[ProblemResult] = field(default_factory=list)  # Slowest 10%

    # Problem complexity analysis
    complexity_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ComprehensiveAccuracyReport:
    """Complete accuracy analysis across all test suites"""
    overall_accuracy: float
    suite_results: Dict[str, SuiteResult]

    # Cross-suite analysis
    best_performing_suites: List[Tuple[str, float]]
    worst_performing_suites: List[Tuple[str, float]]

    # Problem pattern analysis
    problematic_ranges: List[Tuple[int, int, float]]  # (min, max, accuracy)
    reliable_ranges: List[Tuple[int, int, float]]

    # Recommendations
    training_recommendations: List[str]
    architecture_recommendations: List[str]


class AccuracyBenchmarker:
    """
    Comprehensive accuracy benchmarking system for Neural-Klotski.

    Evaluates network performance across systematic test problems,
    analyzes error patterns, and provides optimization recommendations.
    """

    def __init__(self):
        """Initialize accuracy benchmarker"""
        self.config = get_default_config()
        self.task_manager = AdditionTaskManager(self.config.simulation)

        # Define standard test suites
        self.test_suites = self._create_standard_test_suites()

        # Results storage
        self.suite_results: Dict[str, SuiteResult] = {}

    def _create_standard_test_suites(self) -> List[AccuracyTestSuite]:
        """Create standard test suites for comprehensive evaluation"""
        return [
            AccuracyTestSuite(
                name="simple_range",
                description="Simple addition problems (0-15)",
                min_operand=0,
                max_operand=15,
                num_problems=100,
                problem_distribution="uniform"
            ),
            AccuracyTestSuite(
                name="intermediate_range",
                description="Intermediate addition problems (0-127)",
                min_operand=0,
                max_operand=127,
                num_problems=200,
                problem_distribution="uniform"
            ),
            AccuracyTestSuite(
                name="advanced_range",
                description="Advanced addition problems (0-511)",
                min_operand=0,
                max_operand=511,
                num_problems=500,
                problem_distribution="uniform"
            ),
            AccuracyTestSuite(
                name="edge_cases",
                description="Edge case problems (boundaries, zeros, max values)",
                min_operand=0,
                max_operand=511,
                num_problems=50,
                problem_distribution="edge_heavy"
            ),
            AccuracyTestSuite(
                name="small_numbers",
                description="Small number addition (0-31)",
                min_operand=0,
                max_operand=31,
                num_problems=150,
                problem_distribution="gaussian"
            ),
            AccuracyTestSuite(
                name="large_numbers",
                description="Large number addition (256-511)",
                min_operand=256,
                max_operand=511,
                num_problems=150,
                problem_distribution="gaussian"
            ),
            AccuracyTestSuite(
                name="carry_heavy",
                description="Problems requiring many carry operations",
                min_operand=100,
                max_operand=511,
                num_problems=100,
                problem_distribution="uniform"
            ),
            AccuracyTestSuite(
                name="random_comprehensive",
                description="Random sampling across full range",
                min_operand=0,
                max_operand=511,
                num_problems=1000,
                problem_distribution="uniform"
            )
        ]

    def generate_test_problems(self, suite: AccuracyTestSuite) -> List[AdditionProblem]:
        """Generate test problems according to suite specification"""
        problems = []

        if suite.problem_distribution == "uniform":
            # Uniform random distribution
            for _ in range(suite.num_problems):
                a = random.randint(suite.min_operand, suite.max_operand)
                b = random.randint(suite.min_operand, suite.max_operand)
                problems.append(AdditionProblem(a, b, a + b))

        elif suite.problem_distribution == "gaussian":
            # Gaussian distribution centered in range
            center = (suite.min_operand + suite.max_operand) / 2
            std = (suite.max_operand - suite.min_operand) / 6  # 99.7% within range

            for _ in range(suite.num_problems):
                a = int(np.clip(np.random.normal(center, std), suite.min_operand, suite.max_operand))
                b = int(np.clip(np.random.normal(center, std), suite.min_operand, suite.max_operand))
                problems.append(AdditionProblem(a, b, a + b))

        elif suite.problem_distribution == "edge_heavy":
            # Focus on edge cases and boundaries
            edge_values = [
                suite.min_operand, suite.min_operand + 1,
                suite.max_operand - 1, suite.max_operand,
                0, 1, 2, 3, 7, 15, 31, 63, 127, 255, 511
            ]
            edge_values = [v for v in edge_values if suite.min_operand <= v <= suite.max_operand]

            # 70% edge cases, 30% random
            edge_count = int(0.7 * suite.num_problems)
            random_count = suite.num_problems - edge_count

            # Generate edge case problems
            for _ in range(edge_count):
                a = random.choice(edge_values)
                b = random.choice(edge_values)
                problems.append(AdditionProblem(a, b, a + b))

            # Fill with random problems
            for _ in range(random_count):
                a = random.randint(suite.min_operand, suite.max_operand)
                b = random.randint(suite.min_operand, suite.max_operand)
                problems.append(AdditionProblem(a, b, a + b))

        return problems

    def run_test_suite(self, network, suite: AccuracyTestSuite) -> SuiteResult:
        """Run a complete test suite and analyze results"""
        print(f"🧪 Running test suite: {suite.name}")
        print(f"   {suite.description}")
        print(f"   {suite.num_problems} problems, range [{suite.min_operand}, {suite.max_operand}]")

        # Generate test problems
        problems = self.generate_test_problems(suite)

        # Execute test problems
        problem_results = []
        start_time = time.time()

        for i, problem in enumerate(problems):
            if i % 50 == 0 and i > 0:
                print(f"   Progress: {i}/{len(problems)} ({100*i/len(problems):.1f}%)")

            # Run single problem
            network.reset_simulation()
            prob_start = time.time()

            result, stats = self.task_manager.execute_addition_task(network, problem)

            prob_time = time.time() - prob_start

            # Create problem result
            problem_result = ProblemResult(
                problem=problem,
                network_output=result.decoded_sum,
                correct=(result.decoded_sum == problem.expected_sum),
                confidence=result.confidence,
                execution_time=prob_time
            )
            problem_results.append(problem_result)

        total_time = time.time() - start_time

        # Analyze results
        suite_result = self._analyze_suite_results(suite.name, problem_results, total_time)

        print(f"   ✅ Completed: {suite_result.accuracy:.1%} accuracy ({suite_result.correct_answers}/{suite_result.total_problems})")

        return suite_result

    def _analyze_suite_results(self, suite_name: str, problem_results: List[ProblemResult], total_time: float) -> SuiteResult:
        """Analyze results from a test suite"""
        total_problems = len(problem_results)
        correct_answers = sum(1 for r in problem_results if r.correct)
        accuracy = correct_answers / total_problems if total_problems > 0 else 0.0

        # Calculate averages
        avg_confidence = np.mean([r.confidence for r in problem_results])
        avg_execution_time = np.mean([r.execution_time for r in problem_results])

        # Error analysis
        error_distribution = {}
        incorrect_results = [r for r in problem_results if not r.correct]

        for result in incorrect_results:
            error_mag = result.error_magnitude
            error_distribution[error_mag] = error_distribution.get(error_mag, 0) + 1

        # Find worst errors
        worst_errors = sorted(incorrect_results, key=lambda r: r.error_magnitude, reverse=True)[:10]

        # Performance analysis
        sorted_by_time = sorted(problem_results, key=lambda r: r.execution_time)
        fast_count = max(1, len(sorted_by_time) // 10)
        slow_count = max(1, len(sorted_by_time) // 10)

        fast_problems = sorted_by_time[:fast_count]
        slow_problems = sorted_by_time[-slow_count:]

        # Complexity analysis
        complexity_breakdown = self._analyze_problem_complexity(problem_results)

        return SuiteResult(
            suite_name=suite_name,
            total_problems=total_problems,
            correct_answers=correct_answers,
            accuracy=accuracy,
            average_confidence=avg_confidence,
            average_execution_time=avg_execution_time,
            error_distribution=error_distribution,
            worst_errors=worst_errors,
            fast_problems=fast_problems,
            slow_problems=slow_problems,
            complexity_breakdown=complexity_breakdown
        )

    def _analyze_problem_complexity(self, problem_results: List[ProblemResult]) -> Dict[str, Dict[str, float]]:
        """Analyze accuracy by problem complexity"""
        complexity_breakdown = {
            "operand_size": {},
            "sum_size": {},
            "carry_operations": {}
        }

        # Group by operand size ranges
        size_ranges = [(0, 15), (16, 31), (32, 63), (64, 127), (128, 255), (256, 511)]
        for min_val, max_val in size_ranges:
            range_results = [
                r for r in problem_results
                if min_val <= max(r.problem.operand1, r.problem.operand2) <= max_val
            ]
            if range_results:
                accuracy = sum(1 for r in range_results if r.correct) / len(range_results)
                complexity_breakdown["operand_size"][f"{min_val}-{max_val}"] = accuracy

        # Group by sum size ranges
        for min_val, max_val in size_ranges:
            range_results = [
                r for r in problem_results
                if min_val <= r.problem.expected_sum <= max_val
            ]
            if range_results:
                accuracy = sum(1 for r in range_results if r.correct) / len(range_results)
                complexity_breakdown["sum_size"][f"{min_val}-{max_val}"] = accuracy

        # Group by estimated carry operations (rough approximation)
        for carry_count in range(0, 6):
            # Rough heuristic: count bit positions where both operands have 1
            range_results = []
            for r in problem_results:
                estimated_carries = bin(r.problem.operand1 & r.problem.operand2).count('1')
                if estimated_carries == carry_count:
                    range_results.append(r)

            if range_results:
                accuracy = sum(1 for r in range_results if r.correct) / len(range_results)
                complexity_breakdown["carry_operations"][f"{carry_count}"] = accuracy

        return complexity_breakdown

    def run_comprehensive_accuracy_evaluation(self, network=None) -> ComprehensiveAccuracyReport:
        """Run comprehensive accuracy evaluation across all test suites"""
        print("🔬 Running Comprehensive Accuracy Evaluation")
        print("=" * 60)

        if network is None:
            print("Creating fresh network for testing...")
            network = create_addition_network(enable_learning=False)

        overall_start = time.time()

        # Run all test suites
        for suite in self.test_suites:
            try:
                suite_result = self.run_test_suite(network, suite)
                self.suite_results[suite.name] = suite_result
                print()
            except Exception as e:
                print(f"❌ Failed to run suite {suite.name}: {e}")
                print()

        overall_duration = time.time() - overall_start

        # Generate comprehensive report
        report = self._generate_comprehensive_report()

        # Print summary
        self._print_accuracy_summary(report, overall_duration)

        return report

    def _generate_comprehensive_report(self) -> ComprehensiveAccuracyReport:
        """Generate comprehensive accuracy analysis report"""
        if not self.suite_results:
            return ComprehensiveAccuracyReport(
                overall_accuracy=0.0,
                suite_results={},
                best_performing_suites=[],
                worst_performing_suites=[],
                problematic_ranges=[],
                reliable_ranges=[],
                training_recommendations=[],
                architecture_recommendations=[]
            )

        # Calculate overall accuracy (weighted by number of problems)
        total_problems = sum(r.total_problems for r in self.suite_results.values())
        total_correct = sum(r.correct_answers for r in self.suite_results.values())
        overall_accuracy = total_correct / total_problems if total_problems > 0 else 0.0

        # Find best and worst performing suites
        suite_accuracies = [(name, result.accuracy) for name, result in self.suite_results.items()]
        best_suites = sorted(suite_accuracies, key=lambda x: x[1], reverse=True)[:3]
        worst_suites = sorted(suite_accuracies, key=lambda x: x[1])[:3]

        # Analyze problematic and reliable ranges
        problematic_ranges = self._identify_problematic_ranges()
        reliable_ranges = self._identify_reliable_ranges()

        # Generate recommendations
        training_recs = self._generate_training_recommendations()
        architecture_recs = self._generate_architecture_recommendations()

        return ComprehensiveAccuracyReport(
            overall_accuracy=overall_accuracy,
            suite_results=self.suite_results,
            best_performing_suites=best_suites,
            worst_performing_suites=worst_suites,
            problematic_ranges=problematic_ranges,
            reliable_ranges=reliable_ranges,
            training_recommendations=training_recs,
            architecture_recommendations=architecture_recs
        )

    def _identify_problematic_ranges(self) -> List[Tuple[int, int, float]]:
        """Identify number ranges with low accuracy"""
        problematic = []

        # Check ranges from complexity breakdown
        for suite_name, result in self.suite_results.items():
            if "operand_size" in result.complexity_breakdown:
                for range_str, accuracy in result.complexity_breakdown["operand_size"].items():
                    if accuracy < 0.8:  # Less than 80% accuracy
                        min_val, max_val = map(int, range_str.split('-'))
                        problematic.append((min_val, max_val, accuracy))

        # Remove duplicates and sort by accuracy
        unique_problematic = list(set(problematic))
        return sorted(unique_problematic, key=lambda x: x[2])

    def _identify_reliable_ranges(self) -> List[Tuple[int, int, float]]:
        """Identify number ranges with high accuracy"""
        reliable = []

        for suite_name, result in self.suite_results.items():
            if "operand_size" in result.complexity_breakdown:
                for range_str, accuracy in result.complexity_breakdown["operand_size"].items():
                    if accuracy > 0.95:  # Greater than 95% accuracy
                        min_val, max_val = map(int, range_str.split('-'))
                        reliable.append((min_val, max_val, accuracy))

        unique_reliable = list(set(reliable))
        return sorted(unique_reliable, key=lambda x: x[2], reverse=True)

    def _generate_training_recommendations(self) -> List[str]:
        """Generate training optimization recommendations"""
        recommendations = []

        # Check overall performance
        overall_acc = sum(r.accuracy for r in self.suite_results.values()) / len(self.suite_results)

        if overall_acc < 0.7:
            recommendations.append("Overall accuracy low - increase training duration and learning rates")
        elif overall_acc < 0.85:
            recommendations.append("Moderate accuracy - focus on difficult problem ranges")

        # Check for range-specific issues
        if "simple_range" in self.suite_results and self.suite_results["simple_range"].accuracy < 0.9:
            recommendations.append("Poor performance on simple problems - check basic network functionality")

        if "advanced_range" in self.suite_results and self.suite_results["advanced_range"].accuracy < 0.7:
            recommendations.append("Poor performance on complex problems - extend advanced training phase")

        # Check error patterns
        for suite_name, result in self.suite_results.items():
            if result.worst_errors:
                max_error = max(e.error_magnitude for e in result.worst_errors)
                if max_error > 100:
                    recommendations.append(f"Large errors in {suite_name} - review output decoding accuracy")

        if not recommendations:
            recommendations.append("Performance is good - consider testing larger number ranges")

        return recommendations

    def _generate_architecture_recommendations(self) -> List[str]:
        """Generate architecture optimization recommendations"""
        recommendations = []

        # Check execution time patterns
        avg_times = [r.average_execution_time for r in self.suite_results.values()]
        max_time = max(avg_times) if avg_times else 0

        if max_time > 0.1:  # 100ms per problem
            recommendations.append("Slow execution - consider optimizing network size or connectivity")

        # Check confidence patterns
        avg_confidences = [r.average_confidence for r in self.suite_results.values()]
        min_confidence = min(avg_confidences) if avg_confidences else 1.0

        if min_confidence < 0.7:
            recommendations.append("Low confidence scores - review threshold settings and learning parameters")

        # Check error distribution
        total_errors = {}
        for result in self.suite_results.values():
            for error_mag, count in result.error_distribution.items():
                total_errors[error_mag] = total_errors.get(error_mag, 0) + count

        if total_errors and max(total_errors.keys()) > 200:
            recommendations.append("Very large errors detected - check output layer architecture")

        if not recommendations:
            recommendations.append("Architecture performing well - system is properly configured")

        return recommendations

    def _print_accuracy_summary(self, report: ComprehensiveAccuracyReport, duration: float):
        """Print comprehensive accuracy summary"""
        print("\n" + "="*60)
        print("🔬 COMPREHENSIVE ACCURACY EVALUATION SUMMARY")
        print("="*60)

        print(f"Overall accuracy: {report.overall_accuracy:.1%}")
        print(f"Total evaluation time: {duration:.1f}s")
        print(f"Test suites completed: {len(report.suite_results)}")

        # Suite performance
        if report.suite_results:
            print(f"\n📊 Test Suite Performance:")
            for name, result in report.suite_results.items():
                print(f"  • {name}: {result.accuracy:.1%} ({result.correct_answers}/{result.total_problems})")
                print(f"    - Avg confidence: {result.average_confidence:.3f}")
                print(f"    - Avg time: {result.average_execution_time*1000:.1f}ms")

        # Best and worst suites
        if report.best_performing_suites:
            print(f"\n🏆 Best Performing Suites:")
            for name, accuracy in report.best_performing_suites:
                print(f"  • {name}: {accuracy:.1%}")

        if report.worst_performing_suites:
            print(f"\n⚠️ Worst Performing Suites:")
            for name, accuracy in report.worst_performing_suites:
                print(f"  • {name}: {accuracy:.1%}")

        # Problem range analysis
        if report.problematic_ranges:
            print(f"\n🚨 Problematic Ranges:")
            for min_val, max_val, accuracy in report.problematic_ranges:
                print(f"  • Range [{min_val}, {max_val}]: {accuracy:.1%}")

        if report.reliable_ranges:
            print(f"\n✅ Reliable Ranges:")
            for min_val, max_val, accuracy in report.reliable_ranges:
                print(f"  • Range [{min_val}, {max_val}]: {accuracy:.1%}")

        # Recommendations
        if report.training_recommendations:
            print(f"\n💡 Training Recommendations:")
            for i, rec in enumerate(report.training_recommendations, 1):
                print(f"  {i}. {rec}")

        if report.architecture_recommendations:
            print(f"\n🏗️ Architecture Recommendations:")
            for i, rec in enumerate(report.architecture_recommendations, 1):
                print(f"  {i}. {rec}")

        print("="*60)

    def run_trained_network_evaluation(self) -> ComprehensiveAccuracyReport:
        """Train a network and then evaluate its accuracy"""
        print("🎓 Training network for accuracy evaluation...")

        # Create and train network
        config = get_default_config()
        training_config = TrainingConfig()
        training_config.epochs_per_phase = {
            TrainingPhase.SIMPLE: 50,
            TrainingPhase.INTERMEDIATE: 100,
            TrainingPhase.ADVANCED: 200
        }
        training_config.problems_per_epoch = 20

        trainer = AdditionNetworkTrainer(training_config, config.simulation)
        final_metrics = trainer.train()

        print(f"Training completed with {final_metrics.accuracy:.1%} final accuracy")
        print("Running comprehensive accuracy evaluation on trained network...\n")

        # Evaluate trained network
        return self.run_comprehensive_accuracy_evaluation(trainer.network)

    def export_accuracy_results(self, report: ComprehensiveAccuracyReport, filename: str = "accuracy_evaluation.json"):
        """Export accuracy evaluation results to JSON"""
        try:
            import json

            export_data = {
                'overall_accuracy': report.overall_accuracy,
                'suite_results': {},
                'best_performing_suites': report.best_performing_suites,
                'worst_performing_suites': report.worst_performing_suites,
                'problematic_ranges': report.problematic_ranges,
                'reliable_ranges': report.reliable_ranges,
                'training_recommendations': report.training_recommendations,
                'architecture_recommendations': report.architecture_recommendations
            }

            # Export suite results (simplified)
            for name, result in report.suite_results.items():
                export_data['suite_results'][name] = {
                    'accuracy': result.accuracy,
                    'total_problems': result.total_problems,
                    'correct_answers': result.correct_answers,
                    'average_confidence': result.average_confidence,
                    'average_execution_time': result.average_execution_time,
                    'error_distribution': result.error_distribution,
                    'complexity_breakdown': result.complexity_breakdown
                }

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)

            print(f"📁 Accuracy evaluation results exported to {filename}")

        except Exception as e:
            print(f"❌ Failed to export accuracy results: {e}")


if __name__ == "__main__":
    # Test accuracy benchmarker
    print("🧪 Testing Neural-Klotski Accuracy Benchmarker...")

    benchmarker = AccuracyBenchmarker()

    # Run comprehensive evaluation
    report = benchmarker.run_comprehensive_accuracy_evaluation()

    # Export results
    benchmarker.export_accuracy_results(report, "test_accuracy_evaluation.json")

    print("\n✅ Accuracy benchmarking test completed!")