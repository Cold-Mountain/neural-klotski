#!/usr/bin/env python3
"""
Test Neural-Klotski's actual addition capabilities.
Tests single-digit addition and progressively harder problems.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from neural_klotski.core.architecture import create_addition_network
from neural_klotski.core.encoding import AdditionProblem, AdditionTaskManager
from neural_klotski.training.trainer import AdditionNetworkTrainer, TrainingConfig, TrainingPhase
from neural_klotski.config import get_default_config

def test_untrained_network():
    """Test what an untrained network can do"""
    print("🧠 Testing UNTRAINED Network")
    print("=" * 50)

    config = get_default_config()
    network = create_addition_network(enable_learning=False)
    task_manager = AdditionTaskManager(config.simulation)

    # Test simple single-digit problems
    test_problems = [
        (1, 1, 2),
        (2, 3, 5),
        (5, 4, 9),
        (7, 8, 15),
        (9, 9, 18)
    ]

    correct = 0
    for a, b, expected in test_problems:
        problem = AdditionProblem(a, b, expected)
        network.reset_simulation()
        result, stats = task_manager.execute_addition_task(network, problem)

        is_correct = result.decoded_sum == expected
        if is_correct:
            correct += 1

        print(f"  {a} + {b} = {expected} | Network: {result.decoded_sum} | {'✓' if is_correct else '✗'} | Confidence: {result.confidence:.3f}")

    print(f"\nUntrained accuracy: {correct}/{len(test_problems)} = {correct/len(test_problems):.1%}")
    return correct / len(test_problems)

def test_single_digit_addition_complete():
    """Test all single-digit addition combinations (0-9)"""
    print("\n🔢 Testing ALL Single-Digit Addition (0-9)")
    print("=" * 50)

    # First train a network
    config = get_default_config()
    training_config = TrainingConfig()
    training_config.epochs_per_phase = {
        TrainingPhase.SIMPLE: 100,      # Focus on simple problems
        TrainingPhase.INTERMEDIATE: 50,
        TrainingPhase.ADVANCED: 50
    }
    training_config.problems_per_epoch = 30

    print("Training network for single-digit addition...")
    trainer = AdditionNetworkTrainer(training_config, config.simulation)
    final_metrics = trainer.train()
    print(f"Training completed: {final_metrics.accuracy:.1%} final accuracy\n")

    # Test all combinations
    task_manager = AdditionTaskManager(config.simulation)

    total_problems = 0
    correct_problems = 0
    failed_problems = []

    print("Testing all single-digit combinations:")
    print("   ", end="")
    for i in range(10):
        print(f"  {i}", end="")
    print()

    for a in range(10):
        print(f" {a}:", end="")
        for b in range(10):
            expected = a + b
            problem = AdditionProblem(a, b, expected)

            trainer.network.reset_simulation()
            result, stats = task_manager.execute_addition_task(trainer.network, problem)

            is_correct = result.decoded_sum == expected
            total_problems += 1

            if is_correct:
                correct_problems += 1
                print("  ✓", end="")
            else:
                print("  ✗", end="")
                failed_problems.append((a, b, expected, result.decoded_sum))
        print()

    accuracy = correct_problems / total_problems
    print(f"\nSingle-digit accuracy: {correct_problems}/{total_problems} = {accuracy:.1%}")

    if failed_problems:
        print(f"\nFailed problems ({len(failed_problems)}):")
        for a, b, expected, got in failed_problems[:10]:  # Show first 10
            print(f"  {a} + {b} = {expected}, got {got}")
        if len(failed_problems) > 10:
            print(f"  ... and {len(failed_problems) - 10} more")

    return accuracy, len(failed_problems)

def test_progressive_difficulty():
    """Test progressively harder problems"""
    print("\n📈 Testing Progressive Difficulty")
    print("=" * 50)

    # Use the trained network from above
    config = get_default_config()
    training_config = TrainingConfig()
    training_config.epochs_per_phase = {
        TrainingPhase.SIMPLE: 150,
        TrainingPhase.INTERMEDIATE: 150,
        TrainingPhase.ADVANCED: 200
    }
    training_config.problems_per_epoch = 50

    print("Training network for progressive difficulty...")
    trainer = AdditionNetworkTrainer(training_config, config.simulation)
    final_metrics = trainer.train()
    print(f"Training completed: {final_metrics.accuracy:.1%} final accuracy\n")

    task_manager = AdditionTaskManager(config.simulation)

    # Test different ranges
    test_ranges = [
        ("Single digit (0-9)", 0, 9, 20),
        ("Small (0-15)", 0, 15, 25),
        ("Medium (0-31)", 0, 31, 30),
        ("Intermediate (0-63)", 0, 63, 35),
        ("Large (0-127)", 0, 127, 40),
        ("Very large (0-255)", 0, 255, 30)
    ]

    for range_name, min_val, max_val, num_tests in test_ranges:
        correct = 0

        for i in range(num_tests):
            import random
            a = random.randint(min_val, max_val)
            b = random.randint(min_val, max_val)
            expected = a + b

            problem = AdditionProblem(a, b, expected)
            trainer.network.reset_simulation()
            result, stats = task_manager.execute_addition_task(trainer.network, problem)

            if result.decoded_sum == expected:
                correct += 1

        accuracy = correct / num_tests
        print(f"  {range_name:20} {correct:2}/{num_tests:2} = {accuracy:5.1%}")

    return final_metrics.accuracy

def main():
    """Run comprehensive addition capability tests"""
    print("🧪 NEURAL-KLOTSKI ADDITION CAPABILITY TEST")
    print("=" * 60)
    print("Testing the system's ability to perform arithmetic addition...")
    print()

    # Test 1: Untrained network baseline
    untrained_acc = test_untrained_network()

    # Test 2: Complete single-digit addition
    single_digit_acc, failed_count = test_single_digit_addition_complete()

    # Test 3: Progressive difficulty
    progressive_acc = test_progressive_difficulty()

    # Summary
    print("\n" + "=" * 60)
    print("📊 ADDITION CAPABILITY SUMMARY")
    print("=" * 60)
    print(f"Untrained network accuracy:     {untrained_acc:5.1%}")
    print(f"Single-digit addition accuracy: {single_digit_acc:5.1%} ({100-failed_count}/100 correct)")
    print(f"Progressive difficulty accuracy: {progressive_acc:5.1%}")

    print(f"\n🎯 VERDICT:")
    if single_digit_acc >= 0.95:
        print("✅ EXCELLENT - System reliably performs single-digit addition")
    elif single_digit_acc >= 0.80:
        print("🟡 GOOD - System performs most single-digit addition correctly")
    elif single_digit_acc >= 0.50:
        print("🟠 PARTIAL - System has some addition capability but needs improvement")
    else:
        print("❌ LIMITED - System struggles with basic addition")

    if progressive_acc >= 0.80:
        print("✅ System scales well to larger numbers")
    elif progressive_acc >= 0.60:
        print("🟡 System has moderate capability with larger numbers")
    else:
        print("❌ System struggles with larger numbers")

    print(f"\nThe Neural-Klotski system {'IS' if single_digit_acc >= 0.80 else 'IS NOT'} capable of reliable arithmetic addition.")

if __name__ == "__main__":
    main()