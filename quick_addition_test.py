#!/usr/bin/env python3
"""
Quick test of Neural-Klotski's addition capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from neural_klotski.core.architecture import create_addition_network
from neural_klotski.core.encoding import AdditionProblem, AdditionTaskManager
from neural_klotski.training.trainer import AdditionNetworkTrainer, TrainingConfig, TrainingPhase
from neural_klotski.config import get_default_config

def quick_addition_test():
    """Quick test of addition capability"""
    print("🧪 Quick Addition Capability Test")
    print("=" * 40)

    # Test untrained network first
    print("1. Testing untrained network:")
    config = get_default_config()
    network = create_addition_network(enable_learning=False)
    task_manager = AdditionTaskManager(config.simulation)

    test_problems = [(1, 1), (2, 3), (5, 4)]

    for a, b in test_problems:
        expected = a + b
        problem = AdditionProblem(a, b, expected)
        network.reset_simulation()
        result, _ = task_manager.execute_addition_task(network, problem)

        correct = result.decoded_sum == expected
        print(f"   {a} + {b} = {expected} | Network: {result.decoded_sum} | {'✓' if correct else '✗'}")

    # Train a small network
    print("\n2. Training network (minimal):")
    training_config = TrainingConfig()
    training_config.epochs_per_phase = {
        TrainingPhase.SIMPLE: 20,      # Very short training
        TrainingPhase.INTERMEDIATE: 0,
        TrainingPhase.ADVANCED: 0
    }
    training_config.problems_per_epoch = 10

    trainer = AdditionNetworkTrainer(training_config, config.simulation)
    print("   Training...")
    final_metrics = trainer.train()
    print(f"   Training accuracy: {final_metrics.accuracy:.1%}")

    # Test trained network
    print("\n3. Testing trained network:")
    single_digit_tests = [(1, 1), (2, 3), (5, 4), (7, 2), (9, 1), (0, 8)]
    correct_count = 0

    for a, b in single_digit_tests:
        expected = a + b
        problem = AdditionProblem(a, b, expected)
        trainer.network.reset_simulation()
        result, _ = task_manager.execute_addition_task(trainer.network, problem)

        correct = result.decoded_sum == expected
        if correct:
            correct_count += 1
        print(f"   {a} + {b} = {expected} | Network: {result.decoded_sum} | {'✓' if correct else '✗'}")

    accuracy = correct_count / len(single_digit_tests)
    print(f"\n📊 Results:")
    print(f"   Trained network accuracy: {correct_count}/{len(single_digit_tests)} = {accuracy:.1%}")

    if accuracy >= 0.8:
        print("✅ System shows good addition capability")
    elif accuracy >= 0.5:
        print("🟡 System shows partial addition capability")
    else:
        print("❌ System struggles with basic addition")

    return accuracy

if __name__ == "__main__":
    quick_addition_test()